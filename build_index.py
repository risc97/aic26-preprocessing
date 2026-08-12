from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.index import build_index
from config import SHARD_DIR, INDEX_PATH

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=SHARD_DIR)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--bit-width", type=int, default=4, choices=(2, 3, 4),
                        help="4 is the accuracy default; 2 halves the memory")
    args = parser.parse_args()

    try:
        build_index(args.shard_dir, args.index, args.bit_width)
    except (FileNotFoundError, ValueError) as e:
        print(f"FAILED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()
