from __future__ import annotations

import argparse
import sys
from pathlib import Path

from media.videos import find_videos
from pipeline.keyframes import process_video
from config import VIDEOS_DIR, SCENES_DIR, KEYFRAMES_DIR, MEDIA_INFO_DIR, KEYFRAME_MODE

SCENES_SUFFIX = ".scenes.txt"
MEDIA_INFO_SUFFIX = ".json"

DECODER = "libdav1d"  # CPU AV1 decode; containers rarely expose NVDEC
import torch
if torch.cuda.is_available():
    DECODER = "av1_cuvid"  # GPU AV1 decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    parser.add_argument("--keyframes-dir", type=Path, default=KEYFRAMES_DIR)
    parser.add_argument("--media-info-dir", type=Path, default=MEDIA_INFO_DIR)
    parser.add_argument("--decoder", default=DECODER,
                        help=f"ffmpeg decoder")
    parser.add_argument("--quality", type=int, default=2,
                        help="ffmpeg JPEG quality, 2 (best) to 31 (worst)")
    parser.add_argument("--video", nargs='*', metavar="VIDEO_ID",
                        help="only process this video id (repeatable)")
    parser.add_argument("--mode", choices=("left", "mid", "right"),
                        default=KEYFRAME_MODE,
                        help=f"which frame in each scene to take (default: {KEYFRAME_MODE})")
    parser.add_argument("--force", action="store_true",
                        help="re-extract videos that already have keyframes")
    args = parser.parse_args()

    videos = find_videos(args.videos_dir)

    if args.video:
        wanted = set(args.video)
        videos = {vid: path for vid, path in videos.items() if vid in wanted}
    if not videos:
        print(f"no videos found in {args.videos_dir}")
        return 1

    failed = []
    for video_id, video_path in videos.items():
        scenes_file = args.scenes_dir / f"{video_id}{SCENES_SUFFIX}"
        if not scenes_file.is_file():
            print(f"[{video_id}] no scenes at {scenes_file}, skipping")
            failed.append(video_id)
            continue
        media_info_file = args.media_info_dir / f"{video_id}{MEDIA_INFO_SUFFIX}"
        if not process_video(video_id, video_path, scenes_file, media_info_file,
                             args.keyframes_dir, args.decoder, args.quality,
                             args.force, mode=args.mode):
            failed.append(video_id)

    if failed:
        print(f"\n{len(failed)} video(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    main()
