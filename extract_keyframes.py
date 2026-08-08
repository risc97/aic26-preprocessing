from __future__ import annotations

import argparse
import sys
from pathlib import Path

from db import MetadataDatabase
from pipeline.keyframes import process_video

VIDEOS_DIR = Path("data/videos")
SCENES_DIR = Path("data/staging")
KEYFRAMES_DIR = Path("data/keyframes")
MEDIA_INFO_DIR = Path("data/media-info")
DB_PATH = Path("data/metadata.db")

VIDEO_SUFFIX = ".webm"
SCENES_SUFFIX = ".scenes.txt"
MEDIA_INFO_SUFFIX = ".json"
DECODER = "av1_cuvid"  # GPU decode; videos are AV1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    parser.add_argument("--keyframes-dir", type=Path, default=KEYFRAMES_DIR)
    parser.add_argument("--media-info-dir", type=Path, default=MEDIA_INFO_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--decoder", default=DECODER,
                        help=f"ffmpeg decoder (default: {DECODER}; use 'libdav1d' for CPU)")
    parser.add_argument("--quality", type=int, default=2,
                        help="ffmpeg JPEG quality, 2 (best) to 31 (worst)")
    parser.add_argument("--video", action="append", metavar="VIDEO_ID",
                        help="only process this video id (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="re-extract videos that already have keyframes")
    args = parser.parse_args()

    scenes_files = sorted(args.scenes_dir.glob(f"*{SCENES_SUFFIX}"))
    if args.video:
        wanted = set(args.video)
        scenes_files = [f for f in scenes_files
                        if f.name.removesuffix(SCENES_SUFFIX) in wanted]
    if not scenes_files:
        print(f"no {SCENES_SUFFIX} files found in {args.scenes_dir}")
        return 1

    failed = []
    with MetadataDatabase(args.db) as db:
        db.init_schema()
        for scenes_file in scenes_files:
            video_id = scenes_file.name.removesuffix(SCENES_SUFFIX)
            video_path = args.videos_dir / f"{video_id}{VIDEO_SUFFIX}"
            if not video_path.exists():
                print(f"[{video_id}] no video at {video_path}, skipping")
                failed.append(video_id)
                continue
            media_info_file = args.media_info_dir / f"{video_id}{MEDIA_INFO_SUFFIX}"
            if not process_video(db, video_id, video_path, scenes_file, media_info_file,
                                 args.keyframes_dir, args.decoder, args.quality,
                                 args.force):
                failed.append(video_id)

    if failed:
        print(f"\n{len(failed)} video(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    main()
