#!/bin/bash
# Score Clipper 서버 + Cloudflare Tunnel 실행
# 종료: Ctrl+C (서버·터널 모두 정리됨)
# 출력에 나오는 https://xxxx.trycloudflare.com 주소를 밴드 멤버에게 공유하면 된다.
cd "$(dirname "$0")"

command -v yt-dlp >/dev/null || { echo "yt-dlp가 필요합니다: brew install yt-dlp"; exit 1; }
command -v cloudflared >/dev/null || { echo "cloudflared가 필요합니다: brew install cloudflared"; exit 1; }

python3 server.py --port 8765 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

cloudflared tunnel --url http://localhost:8765
