from __future__ import annotations

from pathlib import Path


def read_scenes(scenes_file: Path) -> list[tuple[int, int]]:
    """Read "<frame_start> <frame_end>" lines into a list of spans."""
    spans = []
    with open(scenes_file) as f:
        for line in f:
            if not line.strip():
                continue
            left, right = line.split()
            spans.append((int(left), int(right)))
    return spans
