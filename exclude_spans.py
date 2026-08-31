"""
Drop the advertisement frame ranges

Usage:
    transnetv2_pytorch ../data/videos/ -o ../data/staging/
    python find_ad_span.py --ad L25_V060 1500 1980
    python exclude_spans.py
    python split_slides.py --auto-mask
    python extract_keyframes.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from config import SCENES_DIR
from media.scenes import read_scenes

SCENES_SUFFIX = ".scenes.txt"
ORIGIN_SUFFIX = ".scenes.origin.txt"
SPANS_FILE = SCENES_DIR / "ad_spans.json"


def subtract(span: tuple[int, int], cut: tuple[int, int]) -> list[tuple[int, int]]:
    """Remove frames [cut[0], cut[1]] from an inclusive span."""
    s, e = span
    cs, ce = cut
    if ce < s or cs > e:
        return [span]
    out = []
    if cs > s:
        out.append((s, min(cs - 1, e)))
    if ce < e:
        out.append((max(ce + 1, s), e))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes-dir", type=Path, default=SCENES_DIR)
    p.add_argument("--spans-file", type=Path, default=SPANS_FILE)
    p.add_argument("--video", nargs="*", metavar="VIDEO_ID",
                   help="only process this video id (repeatable)")
    args = p.parse_args()

    ad_spans: dict[str, list[list[int]]] = json.loads(args.spans_file.read_text())
    if args.video:
        wanted = set(args.video)
        ad_spans = {v: s for v, s in ad_spans.items() if v in wanted}

    for video_id, frame_spans in sorted(ad_spans.items()):
        if not frame_spans:
            continue
        scenes_file = args.scenes_dir / f"{video_id}{SCENES_SUFFIX}"
        if not scenes_file.is_file():
            print(f"[{video_id}] no scenes at {scenes_file}, skipping")
            continue

        origin_file = args.scenes_dir / f"{video_id}{ORIGIN_SUFFIX}"
        if not origin_file.is_file():
            origin_file.write_text(scenes_file.read_text())

        cuts = [(int(a), int(b)) for a, b in frame_spans]

        spans = read_scenes(scenes_file)
        n_before = len(spans)
        for cut in cuts:
            next_spans = []
            for span in spans:
                next_spans.extend(subtract(span, cut))
            spans = next_spans

        print(f"[{video_id}] {n_before} -> {len(spans)} scenes "
              f"(dropped ad frame ranges {cuts})")
        scenes_file.write_text("".join(f"{a} {b}\n" for a, b in spans))


if __name__ == "__main__":
    main()
