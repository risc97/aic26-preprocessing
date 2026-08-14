from __future__ import annotations

import shutil
from pathlib import Path

from config import KEYFRAME_MODE
from media.frames import extract_frames
from media.keyframes import find_keyframes
from media.preprocess import has_overlay
from media.scenes import read_scenes, select_frames


def process_video(video_id: str, video_path: Path, scenes_file: Path,
                  media_info_file: Path, keyframes_dir: Path,
                  decoder: str, quality: int, force: bool,
                  mode: str = KEYFRAME_MODE) -> bool:
    """Extract one video's keyframes"""
    kf_dir = keyframes_dir / video_id

    spans = read_scenes(scenes_file)
    if not spans:
        print(f"[{video_id}] no scenes in {scenes_file}, skipping")
        return False

    kept = select_frames(spans, mode)

    if not force and len(find_keyframes(keyframes_dir, video_id)) == len(kept):
        print(f"[{video_id}] keyframes already extracted, skipping")
        return True

    dropped = len(spans) - len(kept)
    if dropped:
        print(f"  skipping {dropped} scene(s) with a duplicate {mode}-frame")

    clean_overlay = has_overlay(media_info_file)
    note = ", blacking out overlays" if clean_overlay else ""
    print(f"[{video_id}] extracting {len(kept)} keyframes{note}...")

    shutil.rmtree(kf_dir, ignore_errors=True)
    try:
        images = extract_frames(video_path, [f for _, f in kept], kf_dir, decoder, quality, clean_overlay)
    except RuntimeError as e:
        print(f"[{video_id}] FAILED: {e}")
        shutil.rmtree(kf_dir, ignore_errors=True)
        return False

    print(f"[{video_id}] saved {len(images)} keyframes to {kf_dir}")
    return True
