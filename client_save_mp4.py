#!/usr/bin/env python3

import asyncio
import websockets
import json
import zlib
import time
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from datetime import datetime

# Import the Decoder class from the model file
from decoder_model import Decoder

SERVER_URI = "ws://localhost:8765"
RECORDING_DURATION = 20  # seconds
FPS = 30  # 예상 FPS

async def save_mp4_client():
    try:
        # Initialize Decoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 사용 중인 디바이스: {device}")
        
        # Load decoder model
        model = Decoder(c=64).to(device)
        model.eval()
        print("📦 디코더 모델 로드 완료")
        
        # VideoWriter 초기화 변수 (첫 프레임 수신 후 설정)
        video_writer = None
        video_width = None
        video_height = None
        
        async with websockets.connect(SERVER_URI, max_size=1_000_000) as websocket:
            print(f"✅ 서버에 연결됨: {SERVER_URI}")
            
            frame_count = 0
            start_time = time.time()
            recording_start_time = time.time()
            decode_times = []
            
            # 출력 파일명 생성 (타임스탬프 포함)
            output_dir = "/app/output"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(output_dir, f"output_{timestamp}.mp4")
            
            print(f"🎬 영상 녹화 시작 (20초간 저장) - 출력 파일: {output_filename}")
            
            while True:
                try:
                    # 메시지 수신
                    message = await websocket.recv()
                    
                    # 헤더 파싱
                    header_len = int.from_bytes(message[:4], 'big')
                    header_json = message[4:4 + header_len].decode('utf-8')
                    header = json.loads(header_json)
                    payload = message[4 + header_len:]
                    
                    # 압축 해제
                    decompressed = zlib.decompress(payload)
                    
                    # 디코딩 시작 시간 측정
                    decode_start = time.time()
                    
                    # Convert to tensor and decode
                    latent_int8 = np.frombuffer(decompressed, dtype=np.int8)
                    latent_float32 = latent_int8.astype(np.float32) * header['scale']
                    latent_tensor = torch.from_numpy(latent_float32).reshape(1, header['c'], header['h'], header['w']).to(device)
                    
                    # Decode the frame
                    with torch.no_grad():
                        output_tensor = model(latent_tensor)
                    
                    # 디코딩 완료 시간 측정
                    decode_end = time.time()
                    decode_time = (decode_end - decode_start) * 1000  # ms
                    decode_times.append(decode_time)
                    
                    # 텐서를 numpy 배열로 변환 (RGB 형식)
                    img_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    # 값 범위를 [0, 255]로 변환하고 uint8로 변환
                    img_rgb = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                    # RGB를 BGR로 변환 (OpenCV 형식)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    
                    # print("img_bgr.shape:", img_bgr.shape, "dtype:", img_bgr.dtype)

            

                    # VideoWriter 초기화 (첫 프레임에서만)
                    if video_writer is None:
                        # 실제 디코딩된 이미지 크기 사용
                        video_height, video_width = img_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            output_filename,
                            fourcc,
                            FPS,
                            (video_width, video_height)
                        )
                        if not video_writer.isOpened():
                            print(f"❌ 비디오 라이터 초기화 실패")
                            break
                        print(f"📹 비디오 라이터 초기화 완료: {video_width}x{video_height} @ {FPS}fps")
                    
                    # 프레임 저장
                    if video_writer is not None and video_writer.isOpened():
                        # 해상도가 변경되었을 경우 프레임 리사이즈
                        if img_bgr.shape[1] != video_width or img_bgr.shape[0] != video_height:
                            img_bgr = cv2.resize(img_bgr, (video_width, video_height))
                        video_writer.write(img_bgr)
                    
                    frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - start_time
                    recording_elapsed = current_time - recording_start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    # 지연시간 계산
                    latency = (current_time - header['timestamp']) * 1000
                    
                    # 평균 디코딩 시간 계산
                    avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
                    
                    remaining_time = RECORDING_DURATION - recording_elapsed
                    print(f"📊 프레임 #{frame_count} | FPS: {fps:.1f} | 지연: {latency:.1f}ms | 크기: {len(payload)} bytes | 디코딩: {decode_time:.1f}ms (평균: {avg_decode_time:.1f}ms) | 남은 시간: {remaining_time:.1f}초")
                    
                    # 20초 경과 시 녹화 종료
                    if recording_elapsed >= RECORDING_DURATION:
                        print(f"⏹️ 녹화 완료 ({RECORDING_DURATION}초)")
                        break
                    
                    # 100프레임마다 통계 출력
                    if frame_count % 100 == 0:
                        avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
                        print(f"📈 총 {frame_count}개 프레임 수신 완료 (평균 FPS: {fps:.1f}, 평균 디코딩: {avg_decode_time:.1f}ms)")
                        
                except websockets.exceptions.ConnectionClosed:
                    print("❌ 서버 연결이 끊어졌습니다.")
                    break
                except Exception as e:
                    print(f"❌ 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # VideoWriter 해제
            if video_writer is not None:
                video_writer.release()
                print(f"💾 영상 저장 완료: {output_filename}")
                print(f"📊 총 {frame_count}개 프레임 저장됨")
                    
    except ConnectionRefusedError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 연결 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎬 영상 녹화 클라이언트 시작...")
    try:
        asyncio.run(save_mp4_client())
    except KeyboardInterrupt:
        print("\n⏹️ 녹화 중단됨")

