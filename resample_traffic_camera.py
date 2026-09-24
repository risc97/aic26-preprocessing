"""
Subdivide N* videos about static traffic camera.

Usages:
    transnetv2_pytorch ../data/videos/ -o ../data/staging/
    python resample_traffic_camera.py
    python extract_keyframes.py
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from config import VIDEOS_DIR, SCENES_DIR
from media.probe import probe_video
from media.scenes import read_scenes
from media.videos import find_videos
from pipeline.ids import KEYFRAMES_PER_VIDEO

SCENES_SUFFIX = ".scenes.txt"
RAW_SUFFIX = ".scenes.raw.txt"

def subdivide(span, stride):
    start, end = span
    length = end - start + 1
    if length < 2 * stride:
        return [span]
    cells = length // stride
    edges = [start + (length * i) // cells for i in range(cells + 1)]
    return [(a, b - 1) for a, b in zip(edges[:-1], edges[1:]) if b - 1 >= a]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    p.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    p.add_argument("--video", action="append", metavar="VIDEO_ID",
                   help="only process this video id (repeatable)")
    p.add_argument("--stride-sec", type=float, default=1.0,
                   help="one scene, and so one keyframe, per this many seconds "
                        "(default: 1)")
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = p.parse_args()

    videos = {v: pth for v, pth in find_videos(args.videos_dir).items() if v.startswith("N")}

    if args.video:
        wanted = set(args.video)
        videos = {v: pth for v, pth in videos.items() if v in wanted}
    if not videos:
        print(f"no N* videos found in {args.videos_dir}")
        return 1

    for video_id, video_path in sorted(videos.items()):
        scenes_file = args.scenes_dir / f"{video_id}{SCENES_SUFFIX}"
        raw_file = args.scenes_dir / f"{video_id}{RAW_SUFFIX}"
        source = raw_file if raw_file.is_file() else scenes_file
        if not source.is_file():
            print(f"[{video_id}] no scenes at {scenes_file}, skipping")
            continue

        spans = read_scenes(source)
        if not spans:
            print(f"[{video_id}] no spans in {source}, skipping")
            continue

        fps, duration_ms = probe_video(video_path)
        stride = max(1, int(round(args.stride_sec * fps)))
        new_spans = [cell for span in spans for cell in subdivide(span, stride)]

        print(f"[{video_id}] {len(spans)} -> {len(new_spans)} scenes ")

        if args.dry_run:
            continue

        if source is scenes_file:
            raw_file.write_text(scenes_file.read_text())
        scenes_file.write_text("".join(f"{a} {b}\n" for a, b in new_spans))
    return 0

if __name__ == "__main__":
    main()
                