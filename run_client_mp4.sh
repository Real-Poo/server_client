#!/bin/bash

# 출력 디렉토리 생성
mkdir -p output

echo "🔧 MP4 저장 클라이언트 Docker 이미지 빌드 중..."
docker build -f Dockerfile_mp4.client -t remote-desktop-client-mp4 .

echo "🚀 MP4 저장 클라이언트 실행 중..."
echo "서버가 실행 중인지 확인하세요 (포트 8765)"
echo "20초간 녹화 후 자동으로 종료됩니다."
echo "출력 파일은 ./output/ 디렉토리에 저장됩니다."
echo ""

docker run --gpus all --rm --network host \
  -v $(pwd)/output:/app/output \
  remote-desktop-client-mp4

