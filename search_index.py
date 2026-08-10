#!/usr/bin/env python3
"""Search keyframes by text query and report the matching images."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pipeline.search import KeyframeSearcher, SearchHit, format_timestamp

DB_PATH = Path("data/metadata.db")
INDEX_PATH = Path("data/index/keyframes.tvim")
CKPT_PATH = Path("data/checkpoints/c2lip.pt")


def report(hits: list[SearchHit], copy_to: Path | None) -> None:
    if not hits:
        print("no results")
        return

    for hit in hits:
        kf = hit.keyframe
        print(f"{hit.rank:3d}. {hit.score:.4f}  {kf.video_id}  "
              f"{format_timestamp(kf.timestamp_ms)}  frame {kf.frame_idx}")
        print(f"      {kf.image_path}")

    if copy_to is None:
        return

    copy_to.mkdir(parents=True, exist_ok=True)
    copied = 0
    for hit in hits:
        src = Path(hit.keyframe.image_path)
        if not src.exists():
            print(f"  missing on disk: {src}")
            continue
        # rank-prefixed so the directory browses in score order
        shutil.copy2(src, copy_to / f"{hit.rank:03d}_{hit.score:.3f}_{src.name}")
        copied += 1
    print(f"\ncopied {copied} image(s) to {copy_to}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*",
                        help="text query; omit for an interactive prompt")
    parser.add_argument("-k", "--top-k", type=int, default=20)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    parser.add_argument("--device", default="cuda",
                        help="cpu is fine here, a query is one short string")
    parser.add_argument("--copy-to", type=Path, metavar="DIR",
                        help="also copy the matched images into DIR")
    args = parser.parse_args()

    for path, what in ((args.index, "index"), (args.db, "database"),
                       (args.ckpt, "checkpoint")):
        if not path.exists():
            print(f"no {what} at {path}")
            return 1

    print(f"loading index and C2LIP onto {args.device}...")
    with KeyframeSearcher(args.index, args.db, args.ckpt, args.device) as searcher:
        if args.query:
            report(searcher.search(" ".join(args.query), args.top_k), args.copy_to)
            return 0

        print("type a query, Ctrl-D to quit")
        while True:
            try:
                query = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if query:
                report(searcher.search(query, args.top_k), args.copy_to)


if __name__ == "__main__":
    sys.exit(main())
