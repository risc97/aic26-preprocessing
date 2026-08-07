from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from media.preprocess import preprocess


def extract_frames(video_path: Path, frame_numbers: list[int], out_dir: Path,
                   decoder: str, quality: int, clean_overlay: bool = False) -> list[Path]:
    """Extract all frame with one ffmpeg call, and clean overlay if need"""
    # filter that grabs every mid-frame at once
    select_filter = "+".join(f"eq(n\\,{fn})" for fn in frame_numbers)

    temp_dir = Path(tempfile.mkdtemp(prefix=f"keyframes_{video_path.stem}_"))
    try:
        subprocess.run([
            "ffmpeg", "-c:v", decoder, "-i", str(video_path),
            "-vf", f"select='{select_filter}'", "-vsync", "0", "-an",
            "-q:v", str(quality), "-y", "-loglevel", "error",
            str(temp_dir / "frame_%05d.jpg"),
        ])

        produced = sorted(temp_dir.glob("frame_*.jpg"))
        if len(produced) != len(frame_numbers):
            raise RuntimeError(
                f"ffmpeg produced {len(produced)} frames, expected {len(frame_numbers)}"
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for i, src in enumerate(produced):
            dst = out_dir / f"{i + 1:04}.jpg"
            shutil.move(str(src), dst)
            if clean_overlay:
                preprocess(dst)
            saved.append(dst)
        return saved
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
