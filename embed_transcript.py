from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from pipeline.transcripts import embed_transcript
from config import TRANSCRIPTS_DIR, SHARD_DIR
from models.gte import Gte

MODEL_TAG = "gte"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-dir", type=Path, default=TRANSCRIPTS_DIR)
    parser.add_argument("--shard-dir", type=Path, default=None,
                        help=f"default: <SHARD_DIR>/{MODEL_TAG}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--video", nargs='*', metavar="VIDEO_ID",
                        help="only encode this video id")
    parser.add_argument("--force", action="store_true",
                        help="re-encode videos that already have a complete shard")
    args = parser.parse_args()

    shard_dir = args.shard_dir or SHARD_DIR / MODEL_TAG
    csv_paths = sorted(args.transcripts_dir.glob("*.csv"))
    if args.video:
        wanted = set(args.video)
        csv_paths = [p for p in csv_paths if p.stem in wanted]
    if not csv_paths:
        print(f"No transcript CSVs found in {args.transcripts_dir}")
        return 1
    print(f"loading {MODEL_TAG} model on {args.device}")
    model = Gte(device=args.device)

    failed = []
    for csv_path in csv_paths:
        if not embed_transcript(model, csv_path, args.batch_size, shard_dir, args.force):
            failed.append(csv_path.stem)

    if failed:
        print(f"\n{len(failed)} video(s) failed: {', '.join(failed)}")
        return 1
    return 0

if __name__ == "__main__":
    main()
