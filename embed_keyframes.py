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

from db import MetadataDatabase
from models import MODEL_CHOICES, VARIANT, VARIANTS, load_encoder
from pipeline.embeddings import embed_video
from config import DB_PATH, SHARD_DIR, CKPT_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="c2lip",
                        help="encoder to embed with")
    parser.add_argument("--variant", default=VARIANT, choices=VARIANTS,
                        metavar="NAME",
                        help=f"siglip2 only: open_clip config (default {VARIANT})")
    parser.add_argument("--shard-dir", type=Path,
                        help="default: <SHARD_DIR>/<model>, so models never overwrite each other's shards")
    parser.add_argument("--ckpt", type=Path,
                        help="c2lip: the authors' checkpoint (default "
                             f"{CKPT_PATH}). siglip2: local weights instead of "
                             "the 'webli' hub tag")
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
    if args.model == "c2lip" and ckpt is None:
        ckpt = CKPT_PATH
    # siglip2 without --ckpt pulls the 'webli' weights from the hub
    if ckpt is not None and not ckpt.exists():
        print(f"missing checkpoint at {ckpt}")
        return 1

    with MetadataDatabase(args.db) as db:
        video_ids = [v.video_id for v in db.videos.list_all(limit=10**9)]
        if args.video:
            wanted = set(args.video)
            video_ids = [v for v in video_ids if v in wanted]
        if args.limit:
            video_ids = video_ids[:args.limit]
        if not video_ids:
            print(f"no videos to encode in {args.db}")
            return 1

        source = ckpt if ckpt is not None else "the 'webli' hub weights"
        print(f"loading {args.model} from {source} onto {args.device}...")
        model = load_encoder(args.model, ckpt=ckpt, variant=args.variant,
                             device=args.device)

        shard_dir = args.shard_dir or SHARD_DIR / model.tag
        print(f"{model.model_name}: dim {model.embed_dim}, shards -> {shard_dir}")

        failed = []
        for video_id in video_ids:
            if not embed_video(db, model, video_id, shard_dir,
                               args.batch_size, args.workers, args.force):
                failed.append(video_id)

    if failed:
        print(f"\n{len(failed)} video(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    main()
