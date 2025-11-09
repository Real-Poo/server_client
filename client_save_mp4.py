#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
import zlib
import json
import cv2
from datetime import datetime
import asyncio
import websockets
import time

from decoder_model import Decoder   # decoder_model.py 안의 Decoder는 train.py와 같은 구조

DECODER_PATH = "models/decoder.pth"
SERVER_URI = "ws://localhost:8765"
RECORDING_DURATION = 20
FPS = 30

async def save_mp4_client():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 사용 중인 디바이스: {device}")

        model = Decoder(c=64).to(device)

        if os.path.exists(DECODER_PATH):
            print(f"📦 학습된 디코더 weight 로드: {DECODER_PATH}")
            state_dict = torch.load(DECODER_PATH, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"⚠️ {DECODER_PATH} 를 찾을 수 없습니다. 랜덤 디코더 사용!")

        model.eval()

        video_writer = None
        video_width = None
        video_height = None

        async with websockets.connect(SERVER_URI, max_size=1_000_000) as websocket:
            print(f"✅ 서버에 연결됨: {SERVER_URI}")

            frame_count = 0
            start_time = time.time()
            recording_start_time = time.time()
            decode_times = []

            output_dir = "/app/output"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(output_dir, f"output_{timestamp}.mp4")

            print(f"🎬 영상 녹화 시작 (20초간 저장) - 출력 파일: {output_filename}")

            while True:
                message = await websocket.recv()

                header_len = int.from_bytes(message[:4], "big")
                header_json = message[4 : 4 + header_len].decode("utf-8")
                header = json.loads(header_json)
                payload = message[4 + header_len :]

                decompressed = zlib.decompress(payload)

                c = header["c"]
                h = header["h"]
                w = header["w"]

                decode_start = time.time()

                latent_int8 = np.frombuffer(decompressed, dtype=np.int8)
                latent_float32 = latent_int8.astype(np.float32) * header["scale"]
                latent_tensor = torch.from_numpy(latent_float32).reshape(1, c, h, w).to(device)

                with torch.no_grad():
                    output_tensor = model(latent_tensor)  # (1,3,H,W), [0,1]

                decode_end = time.time()
                decode_time = (decode_end - decode_start) * 1000
                decode_times.append(decode_time)

                img_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_rgb = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                if video_writer is None:
                    video_height, video_width = img_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        output_filename,
                        fourcc,
                        FPS,
                        (video_width, video_height),
                    )
                    if not video_writer.isOpened():
                        print("❌ 비디오 라이터 초기화 실패")
                        break
                    print(f"📹 비디오 라이터 초기화 완료: {video_width}x{video_height} @ {FPS}fps")

                if img_bgr.shape[1] != video_width or img_bgr.shape[0] != video_height:
                    img_bgr = cv2.resize(img_bgr, (video_width, video_height))
                video_writer.write(img_bgr)

                frame_count += 1
                current_time = time.time()
                elapsed = current_time - start_time
                recording_elapsed = current_time - recording_start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                latency = (current_time - header["timestamp"]) * 1000
                avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
                remaining_time = RECORDING_DURATION - recording_elapsed

                print(
                    f"📊 프레임 #{frame_count} | FPS: {fps:.1f} | 지연: {latency:.1f}ms | "
                    f"크기: {len(payload)} bytes | 디코딩: {decode_time:.1f}ms "
                    f"(평균: {avg_decode_time:.1f}ms) | 남은 시간: {remaining_time:.1f}초"
                )

                if recording_elapsed >= RECORDING_DURATION:
                    print(f"⏹️ 녹화 완료 ({RECORDING_DURATION}초)")
                    break

            if video_writer is not None:
                video_writer.release()
                print(f"💾 영상 저장 완료: {output_filename}")
                print(f"📊 총 {frame_count}개 프레임 저장됨")

    except Exception as e:
        print(f"❌ 연결/실행 오류: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🎬 영상 녹화 클라이언트 시작...")
    asyncio.run(save_mp4_client())
