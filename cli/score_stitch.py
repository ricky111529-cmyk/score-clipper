#!/usr/bin/env python3
"""드럼 커버 영상 하단의 악보 오버레이를 잘라 전체 악보로 이어붙이는 도구.

사용법:
  python score_stitch.py <video> --y 940 --height 140 [--fps 2] [--out outdir]

파이프라인 (페이지 넘김 감지는 웹앱 index.html v4와 같은 알고리즘):
  1. ffmpeg로 fps 간격 프레임에서 악보 띠만 크롭 추출
  2. 픽셀을 잉크 / 종이 / 불확실로 나누고 악보가 없는 프레임(인트로·아웃트로)을 거른다
  3. 사전 스캔으로 크롭에 섞인 영상 행을 찾아 판정에서 뺀다
  4. 페이지 기준 이미지 대비 잉크 변화 픽셀 수로 페이지 넘김 감지 (연속 2표본이 같은 새 페이지일 때 확정)
  5. 페이지마다 최대 7프레임을 65퍼센타일로 합성해 움직이는 마디 하이라이트를 지운다
  6. 세로로 스택 -> full_score.png + A4 페이지네이션 PDF
"""
import argparse
import math
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


# ---------- 페이지 넘김 감지 (웹앱 v4와 동일) ----------
# 픽셀을 잉크(어둡고 무채색) / 종이(밝고 무채색) / 불확실(컬러 하이라이트, 흐린 경계 등)로 나누고,
# 지금 페이지의 기준 이미지와 비교해 "종이였던 곳에 생긴 잉크 + 종이로 바뀐 잉크" 픽셀 수로 판정한다.
# - 하이라이트는 불확실로 빠져서 판정에 영향을 주지 않는다
# - 마디 번호만 다른 페이지도 구분된다 (같은 페이지 잡음 0~13px, 마디 번호만 다른 넘김 79px+ 실측)
# - 전환 잔상은 "연속 2표본이 같은 새 페이지"일 때만 넘김으로 확정해 걸러낸다
# - 크롭에 섞인 영상은 사전 스캔(행 단위)과 움직임 감시(블록 단위)로 판정에서 뺀다
DET_W = 1920          # 판정 해상도(폭): 소스 크기와 무관하게 같은 문턱을 쓴다
INK_MAX, INK_SAT, PAPER_MIN, PAPER_SAT = 110, 60, 170, 40
FLIP_T = 40           # 페이지 넘김 문턱 (픽셀 수)
MIN_RUN = 2           # 2표본 미만으로 끝난 구간은 페이지로 보지 않는다
BLK = 16              # 움직임 감시 블록 크기
PRESCAN_PAIRS = 16    # 사전 스캔 표본 쌍 수
POPCNT = np.array([bin(k).count("1") for k in range(64)], dtype=np.uint8)


class Frame:
    __slots__ = ("i", "ink", "paper", "solid")

    def __init__(self, ink, paper):
        self.i = -1
        self.ink, self.paper = ink, paper
        self.solid = erode(paper)


def load_det(path):
    """판정용 RGB 배열: 크롭을 폭 DET_W로 맞춘다 (1080p 영상이면 원본 해상도 그대로)"""
    img = Image.open(path).convert("RGB")
    if img.width != DET_W:
        img = img.resize((DET_W, max(8, round(img.height * DET_W / img.width))), Image.LANCZOS)
    return np.asarray(img)


def classify(rgb):
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = mx - mn
    ink = (mx < INK_MAX) & (sat < INK_SAT)
    paper = ~ink & (mn > PAPER_MIN) & (sat < PAPER_SAT)
    return ink, paper, float((mx > 200).mean()), float(ink.mean())


def erode(m):
    """3x3 이웃이 전부 True인 곳만 True (가장자리는 False). 1px 흔들림을 차이로 세지 않기 위해 쓴다"""
    out = np.zeros_like(m)
    v = m[:-2] & m[1:-1] & m[2:]
    out[1:-1, 1:-1] = v[:, :-2] & v[:, 1:-1] & v[:, 2:]
    return out


def make_frame(rgb, min_bright, min_ink):
    """악보 띠가 떠 있지 않은 프레임이면 None.

    밝은 배경이 절반 이상(컬러 하이라이트도 밝은 쪽) + 잉크(오선 등)가 조금이라도 있어야 한다.
    축소본이 아니라 판정 해상도 그대로 세야 음표가 드문 페이지도 놓치지 않는다.
    """
    ink, paper, bright, ink_frac = classify(rgb)
    if bright < min_bright or ink_frac < min_ink:
        return None
    return Frame(ink, paper)


def ink_diff(a_ink, a_solid, b_ink, b_solid, mask):
    """a 기준으로 b에 새로 생긴 잉크 + a의 잉크가 b에서 종이가 된 픽셀 수 (mask인 곳만)"""
    return int(np.count_nonzero(b_ink & a_solid & mask) + np.count_nonzero(a_ink & b_solid & mask))


def find_band_rows(pairs, h):
    """사전 스캔: 0.5초 간격 프레임 쌍에서 페이지가 넘어가지 않았는데도 바뀌는 행 = 크롭에 섞인 영상.

    가장 긴 연속 "조용한" 행 구간을 악보 띠로 본다. 반환: ((첫 행, 끝 행), 근거 쌍 수, 크롭 전체 사용 여부)
    """
    act = np.zeros(h, dtype=np.int32)
    used = 0
    for a, b in pairs:
        rows = ((a.ink & b.paper) | (a.paper & b.ink)).sum(axis=1)
        if np.count_nonzero(rows >= 2) >= h * 0.25:
            continue  # 페이지가 넘어간 쌍은 근거로 쓰지 않는다
        used += 1
        act += rows >= 2
    best, s = None, -1
    for y in range(h + 1):
        if y < h and not act[y]:
            if s < 0:
                s = y
        elif s >= 0:
            if best is None or y - s > best[1] - best[0] + 1:
                best = (s, y - 1)
            s = -1
    # 근거가 없거나 띠가 너무 얇게 잡히면 크롭 전체를 쓴다
    if not used or best is None or best[1] - best[0] + 1 < h * 0.2:
        return (0, h - 1), used, True
    return best, used, False


class Run:
    """한 페이지 구간: 표본 번호들과 픽셀별 잉크/종이 득표"""

    def __init__(self, h, w):
        self.members = []
        self.ink_v = np.zeros((h, w), np.uint16)
        self.pap_v = np.zeros((h, w), np.uint16)
        self.ref = None


class Detector:
    """표본을 순서대로 push하면 페이지 구간(Run)으로 나눈다. 악보가 없는 프레임은 None을 넣는다"""

    def __init__(self, w, h, band):
        self.w, self.h = w, h
        self.gw, self.gh = -(-w // BLK), -(-h // BLK)
        # 판정 영역: 악보 띠 행에서 위아래·좌우 가장자리 2px를 뺀 곳
        base = np.zeros((h, w), bool)
        base[max(2, band[0] + 2):min(h - 3, band[1] - 2) + 1, 2:w - 2] = True
        self.base = self.mask = base
        self.bhist = np.zeros((self.gh, self.gw), np.uint8)
        self.last = self.cur = self.pend = None
        self.runs = []
        self.stats = {"max_same": 0, "min_flip": None, "moving_blocks": 0}

    def _observe(self, f):
        """연속 프레임 사이에 잉크<->종이가 자주 뒤집히는 블록(크롭에 섞인 영상 등)은 판정에서 뺀다.

        악보 블록은 페이지가 넘어갈 때만 바뀌므로 최근 6쌍 중 3번 이상 바뀌지 않는다.
        """
        if self.last is not None:
            h, w, gh, gw = self.h, self.w, self.gh, self.gw
            changed = np.zeros((gh * BLK, gw * BLK), np.uint16)
            changed[:h, :w] = (self.last.ink & f.paper) | (self.last.paper & f.ink)
            cnt = changed.reshape(gh, BLK, gw, BLK).sum(axis=(1, 3))
            self.bhist = ((self.bhist << 1) | (cnt >= 2).astype(np.uint8)) & 63
            moving = POPCNT[self.bhist] >= 3
            self.stats["moving_blocks"] = max(self.stats["moving_blocks"], int(moving.sum()))
            self.mask = self.base
            if moving.any():
                # 주변 1블록까지 함께 뺀다
                grow = moving.copy()
                grow[1:] |= moving[:-1]
                grow[:-1] |= moving[1:]
                grow2 = grow.copy()
                grow2[:, 1:] |= grow[:, :-1]
                grow2[:, :-1] |= grow[:, 1:]
                pix = np.repeat(np.repeat(grow2, BLK, axis=0), BLK, axis=1)[:h, :w]
                self.mask = self.base & ~pix
        self.last = f

    def _add_to(self, r, f):
        r.members.append(f.i)
        r.ink_v += f.ink
        r.pap_v += f.paper
        r.ref = None

    def _new_run(self, f):
        r = Run(self.h, self.w)
        self._add_to(r, f)
        return r

    def _ref_of(self, r):
        """페이지 기준 이미지: 지금까지 표본의 잉크/종이 다수결 (하이라이트에 가렸던 곳은 다른 표본이 채운다)"""
        if r.ref is None:
            r.ref = (r.ink_v > r.pap_v, erode(r.pap_v > r.ink_v))
        return r.ref

    def _close(self):
        r, self.cur = self.cur, None
        if r is None or len(r.members) < MIN_RUN:
            return
        prev = self.runs[-1] if self.runs else None
        if prev is not None and prev.ink_v is not None:
            a, b = self._ref_of(prev), self._ref_of(r)
            d = ink_diff(a[0], a[1], b[0], b[1], self.mask)
            if d <= FLIP_T:
                # 영상만 나오는 구간 등으로 같은 페이지가 둘로 끊긴 경우: 합친다
                prev.members.extend(r.members)
                prev.ink_v += r.ink_v
                prev.pap_v += r.pap_v
                prev.ref = None
                return
            mf = self.stats["min_flip"]
            self.stats["min_flip"] = d if mf is None else min(mf, d)
            prev.ink_v = prev.pap_v = prev.ref = None  # 더 이상 비교하지 않으므로 메모리 해제
        self.runs.append(r)

    def push(self, f):
        if f is None:  # 악보가 없는 프레임
            self._close()
            self.pend = self.last = None
            return
        self._observe(f)
        if self.cur is None:
            self.cur = self._new_run(f)
            return
        ref = self._ref_of(self.cur)
        d = ink_diff(ref[0], ref[1], f.ink, f.solid, self.mask)
        if self.pend is not None:
            p = self.pend
            if d <= FLIP_T:
                self.pend = None  # 직전 변화는 일시적이었다 (전환 잔상 등)
            elif ink_diff(p.ink, p.solid, f.ink, f.solid, self.mask) <= FLIP_T:
                # 연속 2표본이 같은 새 페이지: 넘김 확정
                self._close()
                self.cur = self._new_run(p)
                self.pend = None
                self._add_to(self.cur, f)
                return
            else:
                self.pend = f  # 전환이 진행 중: 최신 표본으로 다시 확인
                return
        if d <= FLIP_T:
            self._add_to(self.cur, f)
            self.stats["max_same"] = max(self.stats["max_same"], d)
        else:
            self.pend = f

    def finish(self):
        self._close()
        self.pend = None
        return self.runs


def pick_even(s, e, m):
    length = e - s + 1
    if length <= m:
        return list(range(s, e + 1))
    # 웹앱(Math.round)과 같게 반올림
    return [s + math.floor(i * (length - 1) / (m - 1) + 0.5) for i in range(m)]


def composite_members(members):
    """합성에 쓸 표본: 페이지 구간 양끝(페이드가 걸칠 수 있는 곳)을 잘라내고 최대 7개를 고르게"""
    trim = min(2, len(members) * 20 // 100)
    return [members[j] for j in pick_even(trim, len(members) - 1 - trim, 7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--y", type=int, required=True, help="악보 띠 시작 y좌표(px)")
    ap.add_argument("--height", type=int, required=True, help="악보 띠 높이(px)")
    ap.add_argument("--fps", type=float, default=2.0, help="표본 간격 (초당 프레임 수, 기본 2 = 0.5초)")
    ap.add_argument("--out", default="score_out")
    ap.add_argument("--min-white", type=float, default=0.55, help="악보로 인정할 밝은 영역(RGB 최댓값>200) 최소 비율")
    ap.add_argument("--min-ink", type=float, default=0.002, help="악보로 인정할 잉크 최소 비율")
    args = ap.parse_args()

    out = Path(args.out)
    strips_dir = out / "strips"
    print(f"[1/4] 프레임 추출 중 (fps={args.fps}) ...")
    frames = extract_strips(Path(args.video), args.y, args.height, args.fps, strips_dir)
    print(f"      {len(frames)}장 추출됨")
    if not frames:
        print("프레임을 추출하지 못했습니다. --y/--height 값을 확인하세요.")
        sys.exit(1)

    def read(i):
        f = make_frame(load_det(frames[i]), args.min_white, args.min_ink)
        if f is not None:
            f.i = i
        return f

    print("[2/4] 페이지 넘김 감지 중 ...")
    # 사전 스캔: 영상 전체에서 고르게 고른 이웃 표본 쌍으로 크롭에 섞인 영상 행을 찾는다
    det_h, det_w = load_det(frames[0]).shape[:2]
    n = len(frames)
    pairs = []
    if n >= 2:
        for k in range(PRESCAN_PAIRS):
            i = min(n - 2, round((0.05 + 0.9 * k / (PRESCAN_PAIRS - 1)) * (n - 1)))
            a = read(i)
            b = read(i + 1) if a is not None else None
            if a is not None and b is not None:
                pairs.append((a, b))
    band, used, fallback = find_band_rows(pairs, det_h)

    det = Detector(det_w, det_h, band)
    for i in range(n):
        det.push(read(i))
    runs = det.finish()
    st = det.stats
    print(f"      판정 행 {band[0]}~{band[1]}/{det_h}{' (크롭 전체)' if fallback else ''}, 사전 스캔 근거 {used}쌍")
    print(f"      같은 페이지 최대 차이 {st['max_same']}px, 넘김 최소 차이 {st['min_flip']}px, "
          f"움직임 블록 최대 {st['moving_blocks']}개 (문턱 {FLIP_T}px)")
    print(f"      페이지 {len(runs)}개")

    print("[3/4] 페이지 합성(마디 표시 제거) ...")
    # 페이지마다 최대 7프레임을 합성해 움직이는 마디 하이라이트를 지운다
    pages = []
    for r in runs:
        stack = np.stack([
            np.asarray(Image.open(frames[i]).convert("RGB"), dtype=np.uint8)
            for i in composite_members(r.members)
        ])
        # 65퍼센타일(밝은 쪽): 하이라이트가 표본의 ~60%에 머물러도 지워진다
        pages.append(Image.fromarray(np.percentile(stack, 65, axis=0).astype(np.uint8)))
    if not pages:
        print("악보 페이지를 찾지 못했습니다. --y/--height/--min-white 값을 확인하세요.")
        sys.exit(1)

    print("[4/4] 이어붙이는 중 ...")
    imgs = pages
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

    print(f"완료: {png_path} ({len(pages)}페이지), {pdf_path} ({len(pdf_pages)}쪽)")


if __name__ == "__main__":
    main()
