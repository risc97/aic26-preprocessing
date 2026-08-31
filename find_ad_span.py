"""
Remove all SIU advertisement in L25 batch

Usage:
    python find_ad_span.py --ref-video L25_V060 --ref-start 1500 --ref-end 1980
    # check data/staging/ad_spans.json, then:
    python exclude_spans.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from config import VIDEOS_DIR, SCENES_DIR
from media.probe import probe_video
from media.videos import find_videos
from split_slides import grayscale, DECODER

SPANS_FILE = SCENES_DIR / "ad_spans.json"


def match_scores(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    n, m = len(target), len(ref)
    scores = np.full(max(n - m + 1, 0), np.inf, np.float32)
    ref_i = ref.astype(np.int16)
    for t in range(len(scores)):
        scores[t] = np.abs(target[t:t + m].astype(np.int16) - ref_i).mean()
    return scores


def find_matches(scores: np.ndarray, thresh: float, min_gap: int) -> list[int]:
    below = scores <= thresh
    picked: list[int] = []
    i = 0
    n = len(scores)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j < n and below[j]:
            j += 1
        best = i + int(np.argmin(scores[i:j]))
        if not picked or best - picked[-1] >= min_gap:
            picked.append(best)
        i = j
    return picked


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-video", required=True, metavar="VIDEO_ID")
    p.add_argument("--ref-start", type=int, required=True,
                   help="ad start, as a native frame index in --ref-video")
    p.add_argument("--ref-end", type=int, required=True,
                   help="ad end, as a native frame index in --ref-video")
    p.add_argument("--sample-fps", type=float, default=2.0,
                   help="rescan rate for matching")
    p.add_argument("--max-score", type=float, default=35.0,
                   help="reject matches with mean grey diff above this (default: 35)")
    p.add_argument("--min-gap-sec", type=float, default=None,
                   help="minimum spacing between separate ad occurrences "
                        "(default: length of the reference clip)")
    p.add_argument("--video", nargs="*", metavar="VIDEO_ID",
                   help="only scan these video ids (default: all L25 videos)")
    p.add_argument("--spans-file", type=Path, default=SPANS_FILE)
    args = p.parse_args()

    videos = {stem: path for stem, path in find_videos(VIDEOS_DIR).items()
              if stem.startswith("L25_V")}
    if args.video:
        wanted = set(args.video)
        videos = {v: pth for v, pth in videos.items() if v in wanted}

    ref_path = videos[args.ref_video]
    ref_fps, _ = probe_video(ref_path)
    ref_frames = grayscale(ref_path, args.sample_fps, DECODER)
    lo = int(round(args.ref_start / ref_fps * args.sample_fps))
    hi = int(round(args.ref_end / ref_fps * args.sample_fps)) + 1
    ref_clip = ref_frames[lo:hi]
    if len(ref_clip) < 2:
        raise SystemExit("reference window too short, widen --ref-start/--ref-end")
    print(f"reference clip: {len(ref_clip)} samples ({len(ref_clip) / args.sample_fps:.1f}s)")

    min_gap = (int(round(args.min_gap_sec * args.sample_fps))
               if args.min_gap_sec else len(ref_clip))

    ad_spans: dict[str, list[list[int]]] = {}
    if args.spans_file.is_file():
        ad_spans = json.loads(args.spans_file.read_text())

    for video_id, video_path in sorted(videos.items()):
        fps, _ = probe_video(video_path)
        frames = grayscale(video_path, args.sample_fps, DECODER)
        scores = match_scores(frames, ref_clip)
        starts = find_matches(scores, args.max_score, min_gap)

        new_spans = [[int(round(s / args.sample_fps * fps)),
                      int(round((s + len(ref_clip)) / args.sample_fps * fps))]
                     for s in starts]

        existing = {tuple(s) for s in ad_spans.get(video_id, [])}
        merged = sorted(existing | {tuple(s) for s in new_spans})
        ad_spans[video_id] = [list(s) for s in merged]
        print(f"[{video_id}] {ad_spans[video_id] if ad_spans[video_id] else 'no match'}")

    args.spans_file.write_text(json.dumps(ad_spans, indent=2))
    print(f"\nwrote {args.spans_file}")


if __name__ == "__main__":
    main()
