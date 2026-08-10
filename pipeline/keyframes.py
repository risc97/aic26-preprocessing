from __future__ import annotations

import shutil
from pathlib import Path

from db import MetadataDatabase
from media.frames import extract_frames
from media.preprocess import has_overlay
from media.probe import probe_video
from media.scenes import read_scenes


def process_video(db: MetadataDatabase, video_id: str, video_path: Path,
                  scenes_file: Path, media_info_file: Path, keyframes_dir: Path,
                  decoder: str, quality: int, force: bool) -> bool:
    """Extract one video's keyframes and store data"""
    kf_dir = keyframes_dir / video_id

    if not force and kf_dir.is_dir() and db.keyframes.count_by_video(video_id) > 0:
        print(f"[{video_id}] keyframes already extracted, skipping")
        return True

    spans = read_scenes(scenes_file)
    if not spans:
        print(f"[{video_id}] no scenes in {scenes_file}, skipping")
        return False

    seen: set[int] = set()
    kept: list[tuple[tuple[int, int], int]] = []
    for span in spans:
        mid = (span[0] + span[1]) // 2
        if mid in seen:
            print(f"  skipping scene {span[0]}-{span[1]}: duplicate mid-frame {mid}")
            continue
        seen.add(mid)
        kept.append((span, mid))

    fps, duration_ms = probe_video(video_path)
    clean_overlay = has_overlay(media_info_file)
    note = ", blacking out overlays" if clean_overlay else ""
    print(f"[{video_id}] extracting {len(kept)} keyframes ({fps:.3f} fps){note}...")

    db.videos.upsert(video_id, str(video_path), duration_ms, fps)
    db.keyframes.delete_by_video(video_id)
    db.segments.delete_by_video(video_id)


    shutil.rmtree(kf_dir, ignore_errors=True)
    try:
        images = extract_frames(video_path, [mid for _, mid in kept], kf_dir, decoder,
                                quality, clean_overlay)
    except RuntimeError as e:
        print(f"[{video_id}] FAILED: {e}")
        shutil.rmtree(kf_dir, ignore_errors=True)
        return False

    segments = db.segments.create_many(video_id, [span for span, _ in kept])
    db.keyframes.create_many([
        (video_id, image.stem, mid, int(round(mid / fps * 1000)), str(image), segment.segment_id)
        for (_, mid), segment, image in zip(kept, segments, images)
    ])

    print(f"[{video_id}] saved {len(images)} keyframes and {len(segments)} segments")
    return True
