#!/usr/bin/env python3
"""Create the metadata database and its tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from db import MetadataDatabase

DB_PATH = Path("data/metadata.db")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    with MetadataDatabase(args.db) as db:
        db.init_schema()
    print(f"initialised {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
