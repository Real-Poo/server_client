#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import torch.nn.functional as F

# -----------------------
# 설정
# -----------------------
WIDTH, HEIGHT = 1280, 720        # 실제 서버/클라이언트와 동일하게
CHANNELS = 64                    # latent 채널 수 (Encoder/Decoder의 c)
DATA_DIR = "data/train_frames"   # 학습용 프레임이 들어있는 폴더
MODELS_DIR = "models"
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.pth")
DECODER_PATH = os.path.join(MODELS_DIR, "decoder.pth")

BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# 데이터셋: 실제 프레임 이미지
# -----------------------
class FrameDataset(Dataset):
    def __init__(self, root_dir: str, width: int, height: int):
        self.root_dir = root_dir
        self.width = width
        self.height = height

        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        paths: List[str] = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(root_dir, ext)))
            paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))

        if not paths:
            raise RuntimeError(f"No image files found under {root_dir}")

        self.paths = sorted(paths)
        print(f"📂 Found {len(self.paths)} training frames in {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")

        img_bgr = cv2.resize(img_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img = img_rgb.astype(np.float32) / 255.0  # [0,1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        return img


# -----------------------
# Encoder / Decoder (서버/클라와 동일)
# -----------------------
class Encoder(nn.Module):
    def __init__(self, original_w, original_h, c=CHANNELS):
        super(Encoder, self).__init__()
        self.original_w = original_w
        self.original_h = original_h
        self.c = c

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, c, kernel_size=5, stride=2, padding=2),
        )

    def forward(self, x):
        return self.conv_stack(x)


class Decoder(nn.Module):
    def __init__(self, c=CHANNELS):
        super(Decoder, self).__init__()
        self.c = c

        self.deconv_stack = nn.Sequential(
            nn.ConvTranspose2d(c, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # [0,1]
        )

    def forward(self, x):
        return self.deconv_stack(x)


# -----------------------
# 양자화 Straight-Through Estimator (STE)
# -----------------------
class QuantizeSTE(Function):
    @staticmethod
    def forward(ctx, x, scale):
        # x: float32 latent, scale: scalar (tensor)
        q = torch.round(x / scale).clamp(-128, 127)
        y = q * scale
        # backward에서 x 쪽으로 gradient는 그냥 통과, scale은 학습X
        ctx.save_for_backward(scale)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        scale, = ctx.saved_tensors
        # x 쪽 gradient는 그대로, scale에는 gradient 안줌
        grad_x = grad_output.clone()
        grad_scale = None
        return grad_x, grad_scale


quantize_ste = QuantizeSTE.apply


# -----------------------
# 학습 루프
# -----------------------
def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"🔧 Device: {DEVICE}")
    print(f"📐 Training resolution: {WIDTH}x{HEIGHT}")
    dataset = FrameDataset(DATA_DIR, WIDTH, HEIGHT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    encoder = Encoder(WIDTH, HEIGHT, c=CHANNELS).to(DEVICE)
    decoder = Decoder(c=CHANNELS).to(DEVICE)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=LR)
    criterion = nn.MSELoss()

    global_step = 0

    for epoch in range(NUM_EPOCHS):
        encoder.train()
        decoder.train()

        running_loss = 0.0

        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(DEVICE)  # (B,3,H,W)

            # 1) 인코딩
            latent = encoder(imgs)  # (B,C,H',W')

            # 2) 서버/클라와 동일한 방식의 scale 계산 (배치 전체에 하나의 scale)
            with torch.no_grad():
                scale = latent.detach().abs().amax() / 127.0 + 1e-9  # scalar (tensor)

            # 3) 양자화 + 역양자화 (STE로 gradient는 latent로 통과)
            latent_q = quantize_ste(latent, scale)

            # 4) 디코딩
            recon = decoder(latent_q)

            # 5) 복원 손실
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            global_step += 1

            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / ((batch_idx + 1) * imgs.size(0))
                print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Step {batch_idx+1}/{len(loader)} "
                      f"Loss: {avg_loss:.6f}  scale: {scale.item():.6e}")

        epoch_loss = running_loss / len(dataset)
        print(f"✅ Epoch {epoch+1} finished. Avg Loss: {epoch_loss:.6f}")

        # 에폭마다 임시 저장
        torch.save(encoder.state_dict(), ENCODER_PATH)
        torch.save(decoder.state_dict(), DECODER_PATH)
        print(f"💾 Saved encoder to {ENCODER_PATH}")
        print(f"💾 Saved decoder to {DECODER_PATH}")

    print("🎉 Training finished.")


if __name__ == "__main__":
    train()
