from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# PATCH long dir
def _shorten_tmpdir(max_len: int = 60) -> None:
    """Keep TMPDIR short enough for the dataloader workers' AF_UNIX sockets.

    nix-shell points TMPDIR at a long per-shell directory. multiprocessing puts
    its listener socket at <tmpdir>/pymp-XXXXXXXX/listener-XXXXXXXX, and AF_UNIX
    paths cap at 108 bytes, so a long TMPDIR fails with "AF_UNIX path too long"
    as soon as num_workers > 0.
    """
    if len(tempfile.gettempdir()) <= max_len:
        return
    for candidate in ("/tmp", "/var/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            for var in ("TMPDIR", "TEMP", "TMP", "TEMPDIR"):
                os.environ[var] = candidate
            tempfile.tempdir = candidate
            return


_shorten_tmpdir()

from media.videos import list_video_ids
from models import MODEL_CHOICES
from pipeline.embeddings import embed_video
from config import VIDEOS_DIR, KEYFRAMES_DIR, SHARD_DIR, CKPT_PATH

MODEL = "siglip"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--keyframes-dir", type=Path, default=KEYFRAMES_DIR)
    parser.add_argument("--model", type=str, default=MODEL,
                        choices=list(MODEL_CHOICES),
                        help="encoder to embed with")
    parser.add_argument("--shard-dir", type=Path,
                        help="default: <SHARD_DIR>/<model>, so models never "
                             "overwrite each other's shards")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="path to model checkpoint; omit to use source pretrained weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8,
                        help="dataloader workers for JPEG decode + resize")
    parser.add_argument("--video", action="append", metavar="VIDEO_ID",
                        help="only encode this video id (repeatable)")
    parser.add_argument("--limit", type=int,
                        help="stop after N videos (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="re-encode videos that already have a complete shard")
    args = parser.parse_args()

    ckpt = args.ckpt
    # siglip needs its fine-tuned checkpoint; auto-apply it when omitted
    if ckpt is None and args.model == "siglip" and CKPT_PATH.exists():
        ckpt = CKPT_PATH
    if ckpt is not None and not ckpt.exists():
        print(f"Checkpoint not found at {ckpt}, falling back to source pretrained weights")
        ckpt = None

    shard_dir = args.shard_dir or SHARD_DIR / args.model

    video_ids = list_video_ids(args.videos_dir)
    if args.video:
        wanted = set(args.video)
        video_ids = [v for v in video_ids if v in wanted]
    if args.limit:
        video_ids = video_ids[:args.limit]
    if not video_ids:
        print(f"No videos to encode in {args.videos_dir}")
        return 1

    print(f"loading {args.model} model"
          f"{f' from {ckpt}' if ckpt else ' (source pretrained)'}"
          f" on {args.device}")
    model_cls = MODEL_CHOICES[args.model]
    try:
        model = model_cls(ckpt, device=args.device)
    except ValueError as e:
        print(f"Failed to load {args.model}: {e}")
        return 1

    failed = []
    for video_id in video_ids:
        if not embed_video(model, video_id, args.keyframes_dir, shard_dir,
                           args.batch_size, args.workers, args.force):
            failed.append(video_id)

    if failed:
        print(f"\n{len(failed)} video(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    main()
