from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from media.videos import list_video_ids
from models.owlv2 import Owlv2Detector, MODEL_CHOICES, MODEL_NAME, MODEL_REPOS
from pipeline.detections import detect_video
from config import VIDEOS_DIR, KEYFRAMES_DIR, DETECT_DIR

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--keyframes-dir", type=Path, default=KEYFRAMES_DIR)
    parser.add_argument("--model", default=MODEL_NAME, choices=MODEL_CHOICES)
    parser.add_argument("--shard-dir", type=Path,
                        help="default: <DETECT_DIR>/<model>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--pool", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--video", nargs="*", metavar="VIDEO_ID")
    parser.add_argument("--limit", type=int, help="stop after N videos")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    shard_dir = args.shard_dir or DETECT_DIR / args.model

    video_ids = list_video_ids(args.videos_dir)
    if args.video:
        wanted = set(args.video)
        video_ids = [v for v in video_ids if v in wanted]
    if args.limit:
        video_ids = video_ids[:args.limit]
    if not video_ids:
        print(f"No videos to encode in {args.videos_dir}")
        return 1

    print(f"loading {MODEL_REPOS[args.model]} on {args.device}")
    detector = Owlv2Detector(args.model, device=args.device, top_k=args.top_k,
                             pool=args.pool, nms_iou=args.nms_iou)

    failed = []
    for video_id in video_ids:
        if not detect_video(detector, video_id, args.keyframes_dir, shard_dir,
                            args.batch_size, args.workers, args.force):
            failed.append(video_id)

    if failed:
        print(f"\n{len(failed)} video(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    main()