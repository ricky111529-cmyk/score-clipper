#!/usr/bin/env python3
"""드럼 커버 영상 하단의 악보 오버레이를 잘라 전체 악보로 이어붙이는 도구.

사용법:
  python score_stitch.py <video> --y 940 --height 140 [--fps 2] [--out outdir]

파이프라인:
  1. ffmpeg로 fps 간격 프레임에서 악보 띠만 크롭 추출
  2. 악보가 없는 구간(흰 배경 아님 / 잉크 없음) 필터링
  3. 연속 프레임 차이로 페이지 전환 감지 -> 페이지별 안정 구간의 중간 프레임 선택
  4. 인접/기존 페이지와 중복 제거
  5. 세로로 스택 -> full_score.png + A4 페이지네이션 PDF
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def extract_strips(video, y, height, fps, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    for old in workdir.glob("*.png"):
        old.unlink()
    cmd = [
        "ffmpeg", "-i", str(video),
        "-vf", f"fps={fps},crop=iw:{height}:0:{y}",
        "-y", str(workdir / "%05d.png"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(workdir.glob("*.png"))


def signature(path, thumb_w=320):
    """저해상 그레이스케일 시그니처와 (흰배경 비율, 잉크 비율)을 반환."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    white_ratio = float((arr > 200).mean())
    ink_ratio = float((arr < 100).mean())
    thumb_h = max(8, int(img.height * thumb_w / img.width))
    thumb = np.asarray(img.resize((thumb_w, thumb_h)), dtype=np.float32)
    return thumb, white_ratio, ink_ratio


def diff(a, b):
    return float(np.abs(a - b).mean())


SEGS = 8        # 가로 분할 수
FLIP_SEGS = 5   # 이 수 이상 구간이 바뀌면 페이지 넘김


def seg_diffs(a, b):
    """가로 8구간별 평균 차이. 마디 하이라이트 이동은 1-2구간, 페이지 넘김은 대부분 구간이 바뀐다."""
    d = np.abs(a - b)
    w = d.shape[1]
    bounds = [w * i // SEGS for i in range(SEGS + 1)]
    return [float(d[:, bounds[i]:bounds[i + 1]].mean()) for i in range(SEGS)]


def is_page_flip(a, b, thr):
    # 구간별 문턱은 전역 감도보다 낮게: 음표가 적은 페이지의 넘김도 잡는다
    return sum(x > thr * 0.6 for x in seg_diffs(a, b)) >= FLIP_SEGS


def pick_even(s, e, m):
    length = e - s + 1
    if length <= m:
        return list(range(s, e + 1))
    return [s + round(i * (length - 1) / (m - 1)) for i in range(m)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--y", type=int, required=True, help="악보 띠 시작 y좌표(px)")
    ap.add_argument("--height", type=int, required=True, help="악보 띠 높이(px)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", default="score_out")
    ap.add_argument("--min-white", type=float, default=0.55, help="악보로 인정할 흰 배경 최소 비율")
    ap.add_argument("--min-ink", type=float, default=0.005, help="악보로 인정할 잉크 최소 비율")
    ap.add_argument("--page-diff", type=float, default=6.0, help="이 값 이상 차이나면 새 페이지")
    ap.add_argument("--min-run", type=int, default=2, help="페이지로 인정할 최소 연속 프레임 수")
    args = ap.parse_args()

    out = Path(args.out)
    strips_dir = out / "strips"
    print(f"[1/4] 프레임 추출 중 (fps={args.fps}) ...")
    frames = extract_strips(Path(args.video), args.y, args.height, args.fps, strips_dir)
    print(f"      {len(frames)}장 추출됨")

    print("[2/4] 악보 프레임 분석 중 ...")
    sigs = []
    for f in frames:
        thumb, white, ink = signature(f)
        valid = white >= args.min_white and ink >= args.min_ink
        sigs.append((f, thumb, valid))

    # 안정 구간(같은 페이지가 유지되는 연속 프레임)으로 묶기
    runs = []  # (start_idx, end_idx) inclusive
    run_start = None
    for i, (f, thumb, valid) in enumerate(sigs):
        if not valid:
            if run_start is not None:
                runs.append((run_start, i - 1))
                run_start = None
            continue
        if run_start is None:
            run_start = i
        elif is_page_flip(sigs[i - 1][1], thumb, args.page_diff):
            runs.append((run_start, i - 1))
            run_start = i
    if run_start is not None:
        runs.append((run_start, len(sigs) - 1))

    runs = [(s, e) for s, e in runs if e - s + 1 >= args.min_run]
    print(f"      페이지 후보 {len(runs)}개")

    print("[3/4] 페이지 합성(마디 표시 제거) + 중복 제거 ...")
    # 반복 구절은 악보 내용이 같아도 마디 번호가 다르므로 페이지로 유지해야 한다.
    # 따라서 전체 비교가 아니라 직전 페이지와만 비교해 글리치로 쪼개진 run만 합친다.
    # 페이지마다 최대 7프레임의 픽셀 중앙값을 취해 움직이는 마디 하이라이트를 지운다.
    pages = []  # (Image, thumb)
    for s, e in runs:
        mid = (s + e) // 2
        _, thumb, _ = sigs[mid]
        if pages and not is_page_flip(thumb, pages[-1][1], args.page_diff):
            continue
        stack = np.stack([
            np.asarray(Image.open(sigs[i][0]).convert("RGB"), dtype=np.uint8)
            for i in pick_even(s, e, 7)
        ])
        comp = Image.fromarray(np.median(stack, axis=0).astype(np.uint8))
        pages.append((comp, thumb))
    print(f"      최종 페이지 {len(pages)}개")
    if not pages:
        print("악보 페이지를 찾지 못했습니다. --y/--height/--min-white 값을 확인하세요.")
        sys.exit(1)

    print("[4/4] 이어붙이는 중 ...")
    imgs = [p for p, _ in pages]
    w = imgs[0].width
    gap = 6
    total_h = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
    sheet = Image.new("RGB", (w, total_h), "white")
    y_pos = 0
    for im in imgs:
        sheet.paste(im, (0, y_pos))
        y_pos += im.height + gap
    png_path = out / "full_score.png"
    sheet.save(png_path)

    # A4 세로 비율(1:1.414)로 페이지네이션한 PDF
    per_page = max(1, round((w * 1.414 * 0.92) / (imgs[0].height + gap * 4)))
    pdf_pages = []
    margin = 40
    page_w = w + margin * 2
    page_h = int(page_w * 1.414)
    for chunk_start in range(0, len(imgs), per_page):
        chunk = imgs[chunk_start:chunk_start + per_page]
        page = Image.new("RGB", (page_w, page_h), "white")
        step = (page_h - margin * 2) // per_page
        for j, im in enumerate(chunk):
            page.paste(im, (margin, margin + j * step))
        pdf_pages.append(page)
    pdf_path = out / "full_score.pdf"
    pdf_pages[0].save(pdf_path, save_all=True, append_images=pdf_pages[1:])

    print(f"완료: {png_path} ({len(pages)}페이지 x 4마디), {pdf_path} ({len(pdf_pages)}쪽)")


if __name__ == "__main__":
    main()
