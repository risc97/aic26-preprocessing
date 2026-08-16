from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from models.gte import Gte
from pipeline.embeddings import check_complete_shard, shard_paths
from pipeline.ids import vector_id

def read_rows(path: Path):
    with open(path, newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def embed_transcript(model: Gte, csv_path: Path, batch_size: int, shard_dir: Path, force: bool):
    video_id = csv_path.stem
    rows = read_rows(csv_path)
    if not rows:
        print(f"[{video_id}] empty transcript, skipping")
        return False
    if not force and check_complete_shard(shard_dir, video_id, len(rows), model.EMBED_DIM):
        print(f"[{video_id}] shard already complete ({len(rows)} vectors), skipping")
        return True
    texts = [r["text"] if r["text"].strip() else " " for r in rows]
    print(f"[{video_id}] encoding {len(texts)} transcripts")
    out = np.empty((len(texts), model.EMBED_DIM), dtype = np.float16)
    for i in range(0, len(texts), batch_size):
        vecs = model.encode_texts(texts[i:i+batch_size])
        out[i:i+len(vecs)] = vecs.astype(np.float16)

    ids = np.array([vector_id(video_id, f"{i:04d}") for i in range(len(rows))], dtype=np.int64)

    vec_path, ids_path = shard_paths(shard_dir, video_id)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for path, array in ((vec_path, out), (ids_path, ids)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, array)
        tmp.replace(path)

    print(f"[{video_id}] saved {len(rows)} transcripts embeddings to {vec_path}")
    return True