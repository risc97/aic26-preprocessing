from __future__ import annotations
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm

from media.keyframes import find_keyframes
from pipeline.ids import vector_id

class DetectDataset(Dataset):
    def __init__(self, image_paths: list[str], preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i):
        with Image.open(self.image_paths[i]) as img:
            img = img.convert("RGB")
            return self.preprocess(img), torch.tensor(img.size, dtype=torch.int32)

def shard_paths(shard_dir: Path, video_id: str) -> tuple[Path, Path, Path, Path]:
    # ids  (n,)      int64    packed keyframe id, one row per keyframe
    # emb  (n, K, D) float16  L2-normalised class embeddings
    # box  (n, K, 4) float16  xyxy, normalised to the original keyframe
    # aux  (n, K, 3) float16  logit_shift, logit_scale, objectness
    return (shard_dir / f"{video_id}.ids.npy", shard_dir / f"{video_id}.emb.npy",
            shard_dir / f"{video_id}.box.npy", shard_dir / f"{video_id}.aux.npy")

def check_complete_shard(shard_dir: Path, video_id: str, expected: int, top_k: int, dim: int) -> bool:
    paths = shard_paths(shard_dir, video_id)
    if not all(p.exists() for p in paths):
        return False
    try:
        ids, emb, box, aux = (np.load(p, mmap_mode="r") for p in paths)
    except (ValueError, OSError):
        return False
    return (ids.shape == (expected,) and emb.shape == (expected, top_k, dim)
            and box.shape == (expected, top_k, 4)
            and aux.shape == (expected, top_k, 3))

def detect_video(detector, video_id: str, keyframes_dir: Path, shard_dir: Path,
                 batch_size: int, num_workers: int, force: bool) -> bool:
    keyframes = find_keyframes(keyframes_dir, video_id)
    if not keyframes:
        print(f"[{video_id}] no keyframes in {keyframes_dir / video_id}, skipping")
        return False

    n, K, D = len(keyframes), detector.top_k, detector.embed_dim
    if not force and check_complete_shard(shard_dir, video_id, n, K, D):
        print(f"[{video_id}] detections already complete ({n} keyframes), skipping")
        return True

    print(f"[{video_id}] detecting {len(keyframes)} keyframes...")
    loader = DataLoader(
        DetectDataset([str(p) for p in keyframes], detector.preprocess),
        batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=True, drop_last=False,
    )

    emb = np.empty((n, K, D), dtype=np.float16)
    box = np.empty((n, K, 4), dtype=np.float16)
    aux = np.empty((n, K, 3), dtype=np.float16)
    row = 0
    bar = tqdm(total=n, desc=f"[{video_id}]", unit="frame", mininterval=0.5, leave=False)
    for pixel_values, sizes in loader:
        e, b, a = detector.detect(pixel_values, sizes)
        emb[row:row + len(e)], box[row:row + len(e)], aux[row:row + len(e)] = e, b, a
        row += len(e)
        bar.update(len(e))
    bar.close()
    assert row == n, f"detected {row} of {n}"

    ids = np.array([vector_id(video_id, p.stem) for p in keyframes], dtype=np.int64)
    shard_dir.mkdir(parents=True, exist_ok=True)
    # write to .tmp then rename so a crash never leaves a broken shard
    for path, array in zip(shard_paths(shard_dir, video_id), (ids, emb, box, aux)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, array)
        tmp.replace(path)

    print(f"[{video_id}] saved {n} x {K} boxes to {shard_dir}")
    return True
