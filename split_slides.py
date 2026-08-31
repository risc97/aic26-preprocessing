"""
Split keyframe from L25 batch that contains teaching video.
TransNetV2 have difficulies in detecting shot in a one shot video with a teacher sitting
at a corner with a large screen showing slide he/she is teaching, talking about.

Usages:
    transnetv2_pytorch ../data/videos/ -o ../data/staging/
    python split_slides.py
    python extract_keyframes.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import cv2
import numpy as np
from config import VIDEOS_DIR, SCENES_DIR
from media.probe import probe_video
from media.scenes import read_scenes
from media.videos import find_videos

SCENES_SUFFIX = ".scenes.txt"
RAW_SUFFIX = ".scenes.raw.txt"

W, H = 256, 144

VIDEOS = {
    stem: path
    for stem, path in find_videos(VIDEOS_DIR).items()
    if stem.startswith("L25_V")
}

DECODER = "libdav1d"  # CPU AV1 decode; containers rarely expose NVDEC
import torch
if torch.cuda.is_available():
    DECODER = "av1_cuvid"  # GPU AV1 decode

def grayscale(video_path: Path, sample_fps: float, decoder: str | None):
    """Grayscale the video"""
    cmd = [
        "ffmpeg", "-nostdin", "-c:v", decoder, "-i", str(video_path),
        "-vf", f"fps={sample_fps},scale={W}:{H}",
        "-pix_fmt", "gray", "-f", "rawvideo", "-an",
        "-loglevel", "error", "pipe:1",
    ]
    nbytes = W * H
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames = [] 
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if buf is None or len(buf) < nbytes:
                break
            frames.append(np.frombuffer(buf, np.uint8).reshape(H, W))
    finally:
        proc.stdout.close()
        proc.wait()
    if not frames:
        return np.empty((0, H, W), np.uint8)
    return np.stack(frames)

def build_mask(frames: np.ndarray, motion_pct: float):
    """
    Mask the presenter, to avoid checking the motion of the presenter
    True = mask
    """
    mask = np.ones((H, W), bool)

    motion = np.zeros((H, W), np.float32)
    for i in range(1, len(frames)):
        motion += cv2.absdiff(frames[i], frames[i - 1])
    motion /= len(frames) - 1

    cut = max(float(np.percentile(motion, 100.0 - motion_pct)), 1.5)
    busy = (motion > cut).astype(np.uint8)
    busy = cv2.dilate(busy, np.ones((5, 5), np.uint8), iterations=1)
    mask &= busy == 0

    if mask.sum() < 0.05 * W * H:
        mask = np.ones((H, W), bool)
    return mask

def change_scores(frames: np.ndarray, mask: np.ndarray, pixel_thresh: int, block: int = 16):
    n = len(frames)
    scores = np.zeros(n, np.float32)

    pad_y, pad_x = (-H) % block, (-W) % block
    gy, gx = (H + pad_y) // block, (W + pad_x) // block

    # reshape [N, W] array into [gy, gx] grid of block
    def to_blocks(a: np.ndarray) -> np.ndarray:
        return np.pad(a, ((0, pad_y), (0, pad_x))).reshape(
            gy, block, gx, block).sum(axis=(1, 3))

    mask_blk = to_blocks(mask).astype(np.float32)
    valid = mask_blk >= 20  # ignore blocks the mask has emptied out
    if not valid.any():
        return scores

    # calculate the changing score within pairs
    prev = cv2.GaussianBlur(frames[0], (3, 3), 0)
    for i in range(1, n):
        cur = cv2.GaussianBlur(frames[i], (3, 3), 0)
        changed = (cv2.absdiff(cur, prev) > pixel_thresh) & mask
        scores[i] = (to_blocks(changed)[valid] / mask_blk[valid]).max()
        prev = cur
    return scores

def peaks(scores: np.ndarray, thresh: float, min_gap: int):
    above = scores > thresh
    picked: list[int] = []
    i = 1
    n = len(scores)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:  # a fade/wipe spans several samples
            j += 1
        best = i + int(np.argmax(scores[i:j]))
        if not picked or best - picked[-1] >= min_gap:
            picked.append(best)
        i = j
    return picked

def split_span(span: tuple[int, int], cuts: list[int], min_len: int):
    start, end = span
    out = []
    prev = start
    for c in cuts:
        if c - prev < min_len or end - c < min_len:
            continue
        out.append((prev, c - 1))
        prev = c
    out.append((prev, end))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    p.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    p.add_argument("--video", nargs='*', metavar="VIDEO_ID",
                            help="only split_slides this video id")
    p.add_argument("--sample-fps", type=float, default=2.0,
                       help="rescan rate; raise for tighter boundaries")
    p.add_argument("--min-scene-sec", type=float, default=1.5,
                   help="never emit a sub-scene shorter than this (default: 1.5)")
    p.add_argument("--min-split-sec", type=float, default=25.0,
                   help="only sub-split shots longer than this"
                        "(default: 25)")
    p.add_argument("--max-scene-sec", type=float, default=30.0,
                   help="hard-split anything still longer than this (default: 30)")
    p.add_argument("--motion-pct", type=float, default=20.0,
                       help="percent of busiest pixels (default: 20)")
    p.add_argument("--pixel-thresh", type=int, default=14,
                       help="per-pixel grey delta counted as changed (default: 14)")
    p.add_argument("--block-size", type=int, default=16,
                       help="block size for detecting localised changes (default: 16)")
    p.add_argument("--threshold", type=float, default=0.10,
                       help="fraction of the most-changed block that must change (default: 0.10)")
    args = p.parse_args()

    videos = VIDEOS
    if args.video:
        wanted = set(args.video)
        videos = {v: pth for v, pth in videos.items() if v in wanted}

    sample_fps = args.sample_fps
    motion_pct = args.motion_pct
    pixel_thresh = args.pixel_thresh
    block_size = args.block_size
    threshold = args.threshold
    for video_id, video_path in sorted(videos.items()):
        scenes_file = args.scenes_dir / f"{video_id}{SCENES_SUFFIX}"
        raw_file = args.scenes_dir / f"{video_id}{RAW_SUFFIX}"
        source = raw_file if raw_file.is_file() else scenes_file
        if not source.is_file():
            print(f"[{video_id}] no scenes at {scenes_file}, skipping")
            continue

        spans = read_scenes(source)
        fps, duration_ms = probe_video(video_path)

        frames = grayscale(video_path, sample_fps, DECODER)
        if len(frames) < 2:
            print(f"[{video_id}] decode produced {len(frames)} samples, skipping")
            continue

        min_gap = max(1, int(round(args.min_scene_sec * sample_fps)))
        min_len = max(1, int(round(args.min_scene_sec * fps)))
        min_split = int(round(args.min_split_sec * fps))
        max_len = int(round(args.max_scene_sec * fps))

        new_spans: list[tuple[int, int]] = [] # sub-splited span
        n_cuts = 0

        for span in spans:
            if span[1] - span[0] + 1 < min_split:
                new_spans.append(span)
                continue

            lo = int(round(span[0] / fps * sample_fps))
            hi = int(round(span[1] / fps * sample_fps)) + 1
            window = frames[lo:hi]

            # mask the presenter
            mask = build_mask(window, motion_pct)
            scores = change_scores(window, mask, pixel_thresh, block_size)

            cuts = [int(round((lo + i - 0.5) * fps / sample_fps))
                    for i in peaks(scores, threshold, min_gap)]
            cuts = [c for c in cuts if span[0] < c <= span[1]]
            n_cuts += len(cuts)

            new_spans.extend(split_span(span, cuts, min_len))

        print(f"[{video_id}] {len(spans)} -> {len(new_spans)} scenes ({n_cuts} slide cuts)")

        args.scenes_dir.mkdir(parents=True, exist_ok=True)
        if source is scenes_file and not raw_file.is_file():
            raw_file.write_text(scenes_file.read_text())
        (args.scenes_dir / f"{video_id}{SCENES_SUFFIX}").write_text(
            "".join(f"{a} {b}\n" for a, b in new_spans))



if __name__ == "__main__":
    main()