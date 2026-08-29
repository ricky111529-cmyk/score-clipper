#!/usr/bin/env python3
"""Score Clipper 서버: 웹앱 서빙 + 유튜브 다운로드 API.

파이썬 표준 라이브러리만 사용. 외부 의존성은 yt-dlp CLI(+ffmpeg)뿐.
Cloudflare Tunnel 등으로 공개하면 밴드 멤버가 유튜브 링크만으로 악보를 뽑을 수 있다.

    python3 server.py --port 8765

API:
    GET  /api/health          서버 살아있는지 (웹앱이 이걸로 링크 입력칸 표시 여부 결정)
    POST /api/fetch {url}     유튜브 다운로드 시작 -> {job}
    GET  /api/fetch/<job>     진행 상태 {status, pct, file, title, error}
    GET  /api/file/<name>     받아둔 영상 파일 전달
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"
YOUTUBE_RE = re.compile(r"^https?://(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/\S+$")
MAX_DURATION = 20 * 60          # 20분 초과 영상 거부
MAX_CACHE_BYTES = 5 << 30       # 캐시 5GB 초과 시 오래된 것부터 삭제
JOBS = {}                       # job_id -> dict
LOCK = threading.Lock()


def cleanup_cache():
    files = sorted(CACHE.glob("*.mp4"), key=lambda f: f.stat().st_mtime)
    for f in files:
        if time.time() - f.stat().st_mtime > 24 * 3600:
            f.unlink(missing_ok=True)
    files = [f for f in files if f.exists()]
    total = sum(f.stat().st_size for f in files)
    while files and total > MAX_CACHE_BYTES:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)


def set_job(job_id, **kw):
    with LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


def download(job_id, url):
    try:
        out = CACHE / f"{job_id}.mp4"
        if out.exists():
            set_job(job_id, status="done", pct=100, file=out.name)
            return
        # 길이/제목 확인 (봇 차단·비공개 영상도 여기서 걸러짐)
        probe = subprocess.run(
            ["yt-dlp", "--no-playlist", "--print", "%(duration)s\t%(title)s", url],
            capture_output=True, text=True, timeout=90,
        )
        if probe.returncode != 0:
            raise RuntimeError("영상 정보를 가져오지 못했어요. 링크를 확인해 주세요.")
        duration_s, _, title = probe.stdout.strip().partition("\t")
        if duration_s.replace(".", "").isdigit() and float(duration_s) > MAX_DURATION:
            raise RuntimeError("20분이 넘는 영상은 지원하지 않아요.")
        set_job(job_id, title=title[:120], status="downloading", pct=0)

        tmp = CACHE / f"{job_id}.part.mp4"
        proc = subprocess.Popen(
            ["yt-dlp", "--no-playlist", "--newline",
             "-f", "bv*[vcodec^=avc1][height<=1080]/bv*[ext=mp4][height<=1080]/b[ext=mp4]/b",
             "--merge-output-format", "mp4",
             "-o", str(tmp), url],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            m = re.search(r"(\d+(?:\.\d+)?)%", line)
            if m:
                set_job(job_id, pct=float(m.group(1)))
        if proc.wait() != 0 or not tmp.exists():
            raise RuntimeError("다운로드 실패. 잠시 후 다시 시도해 주세요.")
        tmp.rename(out)
        set_job(job_id, status="done", pct=100, file=out.name)
    except Exception as e:
        set_job(job_id, status="error", error=str(e))


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._file(ROOT / "index.html", "text/html; charset=utf-8")
        if p == "/api/health":
            return self._json({"ok": True})
        if p.startswith("/api/fetch/"):
            job = JOBS.get(p.rsplit("/", 1)[1])
            return self._json(job or {"status": "unknown"}, 200 if job else 404)
        if p.startswith("/api/file/"):
            name = p.rsplit("/", 1)[1]
            if re.fullmatch(r"[0-9a-f]{16}\.mp4", name) and (CACHE / name).exists():
                return self._file(CACHE / name, "video/mp4")
            return self._json({"error": "not found"}, 404)
        # 로컬 테스트용: 프로젝트 루트의 mp4 (git에는 올라가지 않음)
        if re.fullmatch(r"/[\w.-]+\.mp4", p) and (ROOT / p[1:]).exists():
            return self._file(ROOT / p[1:], "video/mp4")
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.split("?")[0] != "/api/fetch":
            return self._json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            url = str(body.get("url", "")).strip()
        except Exception:
            return self._json({"error": "잘못된 요청"}, 400)
        if not YOUTUBE_RE.match(url):
            return self._json({"error": "유튜브 링크만 지원해요."}, 400)
        if shutil.which("yt-dlp") is None:
            return self._json({"error": "서버에 yt-dlp가 없어요. brew install yt-dlp"}, 500)
        cleanup_cache()
        job_id = hashlib.sha1(url.encode()).hexdigest()[:16]
        with LOCK:
            running = job_id in JOBS and JOBS[job_id].get("status") in ("queued", "downloading")
        if not running:
            set_job(job_id, status="queued", pct=0, error=None, file=None)
            threading.Thread(target=download, args=(job_id, url), daemon=True).start()
        return self._json({"job": job_id})

    def log_message(self, fmt, *args):  # 요청 로그 간소화
        if "/api/fetch" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    cleanup_cache()
    print(f"Score Clipper 서버: http://localhost:{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
