from __future__ import annotations

from pathlib import Path

import numpy as np

INDEX_SUFFIX = ".tvim"


def list_shards(shard_dir: Path) -> list[tuple[Path, Path]]:
    """(vectors, ids) shard pairs, sorted by video id."""
    pairs = []
    for vec_path in sorted(shard_dir.glob("*.npy")):
        if vec_path.name.endswith(".ids.npy"):
            continue
        ids_path = vec_path.with_suffix(".ids.npy")
        if ids_path.exists():
            pairs.append((vec_path, ids_path))
    return pairs


def _load_shard(vec_path: Path, ids_path: Path, dim: int) -> tuple[np.ndarray, np.ndarray]:
    vecs = np.ascontiguousarray(np.load(vec_path), dtype=np.float32)
    ids = np.load(ids_path).astype(np.uint64)
    if vecs.shape[0] != ids.shape[0]:
        raise ValueError(f"{vec_path.name}: {vecs.shape[0]} vectors vs {ids.shape[0]} ids")
    if vecs.shape[1] != dim:
        raise ValueError(f"{vec_path.name}: dim {vecs.shape[1]}, expected {dim}")
    return vecs, ids

def build_index(shard_dir: Path, index_path: Path, bit_width: int = 4) -> int:
    from turbovec import IdMapIndex

    shards = list_shards(shard_dir)
    if not shards:
        raise FileNotFoundError(f"no embedding shards in {shard_dir}")
    print(f"found {len(shards)} shard(s) in {shard_dir}")

    dim = int(np.load(shards[0][0], mmap_mode="r").shape[1])
    print(f"embedding dim {dim} (from {shards[0][0].name})")

    index = IdMapIndex(dim=dim, bit_width=bit_width)

    chunks, id_chunks, total = [], [], 0
    for vec_path, ids_path in shards:
        vecs, ids = _load_shard(vec_path, ids_path, dim)
        chunks.append(vecs)
        id_chunks.append(ids)
        total += len(ids)
        print(f"  loaded {vec_path.stem}: {len(ids)} ({total} total)")

    vectors = np.concatenate(chunks)
    ids = np.concatenate(id_chunks)
    del chunks, id_chunks

    print(f"adding {total} vectors in one call...")
    index.add_with_ids(vectors, ids)

    index.prepare()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.write(str(index_path))
    print(f"wrote {total} vectors ({bit_width}-bit) to {index_path}")
    return total
