from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import resolve_stored_path
from db import MetadataDatabase
from models import C2Lip, EMBED_DIM
from pipeline.ids import vector_id


class KeyframeDataset(Dataset):
    def __init__(self, image_paths: list[str], preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.image_paths[i]) as img:
            return self.preprocess(img.convert("RGB"))


def shard_paths(shard_dir: Path, video_id: str) -> tuple[Path, Path]:
    # <shard_dir>/<video_id>.npy      float16 (n, 768), L2-normalised
    # <shard_dir>/<video_id>.ids.npy  int64   (n,)      packed vector id per row
    return shard_dir / f"{video_id}.npy", shard_dir / f"{video_id}.ids.npy"


def check_complete_shard(shard_dir: Path, video_id: str, expected: int) -> bool:
    vec_path, ids_path = shard_paths(shard_dir, video_id)
    if not (vec_path.exists() and ids_path.exists()):
        return False
    try:
        vecs = np.load(vec_path, mmap_mode="r")
        ids = np.load(ids_path, mmap_mode="r")
    except (ValueError, OSError):
        return False
    return vecs.shape == (expected, EMBED_DIM) and ids.shape == (expected,)


def embed_video(db: MetadataDatabase, model: C2Lip, video_id: str, shard_dir: Path,
                batch_size: int, num_workers: int, force: bool) -> bool:
    """Encode one video's keyframes into a shard"""
    keyframes = db.keyframes.list_by_video(video_id)
    if not keyframes:
        print(f"[{video_id}] no keyframes in db, skipping")
        return False

    if not force and check_complete_shard(shard_dir, video_id, len(keyframes)):
        print(f"[{video_id}] shard already complete ({len(keyframes)} vectors), skipping")
        return True

    images = [resolve_stored_path(k.image_path) for k in keyframes]
    missing = [p for p in images if not p.exists()]
    if missing:
        print(f"[{video_id}] FAILED: {len(missing)} keyframe image(s) missing, "
              f"first is {missing[0]}")
        return False

    print(f"[{video_id}] encoding {len(keyframes)} keyframes...")
    loader = DataLoader(
        KeyframeDataset([str(p) for p in images], model.preprocess),
        batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=True, drop_last=False,
    )

    out = np.empty((len(keyframes), EMBED_DIM), dtype=np.float16)
    row = 0
    for batch in loader:
        vecs = model.encode_images(batch)
        out[row:row + len(vecs)] = vecs.astype(np.float16)
        row += len(vecs)
    assert row == len(keyframes), f"encoded {row} of {len(keyframes)}"

    ids = np.array([vector_id(k.video_id, k.keyframe_id) for k in keyframes], dtype=np.int64)
    vec_path, ids_path = shard_paths(shard_dir, video_id)
    shard_dir.mkdir(parents=True, exist_ok=True)
    # write to .tmp then rename so a crash never create a broken shard
    for path, array in ((vec_path, out), (ids_path, ids)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, array)
        tmp.replace(path)

    print(f"[{video_id}] saved {row} embeddings to {vec_path}")
    return True
