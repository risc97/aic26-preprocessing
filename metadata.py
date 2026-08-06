from __future__ import annotations
 
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Generator, Optional

@dataclass
class Video:
    video_id: str
    video_path: str
    duration_ms: int
    fps: float

@dataclass
class Keyframe:
    keyframe_id: int
    video_id: str
    frame_id: int
    timestamp_ms: int
    image_path: str
    segment_id: int

@dataclass
class Segment:
    segment_id: int
    video_id: str
    frame_start: int
    frame_end: int

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id      TEXT PRIMARY KEY,
    video_path    TEXT NOT NULL,
    duration_ms   INTEGER,
    fps           REAL
);
 
CREATE TABLE IF NOT EXISTS segments (
    segment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    frame_start   INTEGER NOT NULL,
    frame_end     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);
 
CREATE TABLE IF NOT EXISTS keyframes (
    keyframe_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    frame_id     INTEGER NOT NULL,
    timestamp_ms  INTEGER NOT NULL,
    image_path    TEXT NOT NULL,
    segment_id    INTEGER REFERENCES segments(segment_id) ON DELETE SET NULL,
    UNIQUE(video_id, frame_id)
);
CREATE INDEX IF NOT EXISTS idx_keyframes_video ON keyframes(video_id);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_time ON keyframes(video_id, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_id ON keyframes(video_id, frame_id);
"""

class MetadataDatabase:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self) -> "MetadataDatabase":
        return self
 
    def __exit__(self, *exc) -> None:
        self.close()
 
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # Video CRUD

    def create_video(self, video_id: str, video_path: str, duration_ms: int, fps: float) -> Video:
        self.conn.execute(
            "INSERT INTO videos (video_id, video_path, duration_ms, fps) VALUES (?, ?, ?, ?)",
            (video_id, video_path, duration_ms, fps),
        )
        self.conn.commit()
        return Video(video_id, video_path, duration_ms, fps)

    def get_video(self, video_id: str) -> Video | None:
        row = self.conn.execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        return _row_to_video(row) if row else None

    def list_video(self, limit: int = 1000, offset: int = 0) -> list[Video]:
        rows = self.conn.execute(
            "SELECT * FROM videos ORDER BY video_id LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        return [_row_to_video(r) for r in rows]

    def update_video(
        self, video_id: str, video_path: str | None = None,
        duration_ms: int | None = None, fps: float | None = None,
    ) -> Video | None:
        existing = self.get_video(video_id)
        if existing is None:
            return None
        video_path = existing.video_path if video_path is None else video_path
        duration_ms = existing.duration_ms if duration_ms is None else duration_ms
        fps = existing.fps if fps is None else fps
        self.conn.execute(
            "UPDATE videos SET video_path = ?, duration_ms = ?, fps = ? WHERE video_id = ?",
            (video_path, duration_ms, fps, video_id),
        )
        self.conn.commit()
        return Video(video_id, video_path, duration_ms, fps)

    def delete_video(self, video_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # Segment CRUD

    def create_segment(self, video_id: str, frame_start: int, frame_end: int) -> Segment:
        cur = self.conn.execute(
            "INSERT INTO segments (video_id, frame_start, frame_end) VALUES (?, ?, ?)",
            (video_id, frame_start, frame_end),
        )
        self.conn.execute()
        return Segment(cur.lastrowid, video_id, frame_start, frame_end)

    def get_segment(self, segment_id: int) -> Segment | None:
        row = self.conn.execute(
            "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        return _row_to_segment(row) if row else None

    def get_segments_by_video(self, video_id: str) -> list[Segment]:
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE video_id = ?", (video_id,)
        ).fetchall()
        return [_row_to_segment(row) for row in rows]

    def update_segment(self, segment_id: int, frame_start: int, frame_end: int) -> Segment | None:
        existing = self.get_segment(segment_id)
        if existing is None:
            return None
        frame_start = existing.frame_start if frame_start is None else frame_start
        frame_end = existing.frame_end if frame_end is None else frame_end
        self.conn.execute(
            "UPDATE segments SET frame_start = ?, frame_end = ? WHERE segment_id = ?",
            (frame_start, frame_end, segment_id),
        )
        self.conn.commit()
        return Segment(segment_id, existing.video_id, frame_start, frame_end)

    def delete_segment(self, segment_id: int):
        cur = self.conn.execute(
            "DELETE FROM segments WHERE segment_id = ?",
            (segment_id,),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # Keyframe CRUD

    def create_keyframe(self, video_id: str, frame_id: int, timestamp_ms: int, image_path: str, segment_id: int) -> Keyframe:
        cur = self.conn.execute(
            "INSERT INTO keyframes (video_id, frame_idx, timestamp_ms, image_path, scene_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (video_id, frame_id, timestamp_ms, image_path, segment_id),
        )
        self.conn.commit()
        return Keyframe(cur.lastrowid, video_id, frame_id, timestamp_ms, image_path, segment_id)

    def get_keyframe(self, keyframe_id: int) -> Keyframe | None:
        row = self.conn.execute(
            "SELECT * FROM keyframes WHERE keyframe_id = ?", (keyframe_id,)
        ).fetchone()
        return _row_to_keyframe(row) if row else None

    def get_keyframes_by_ids(self, ids: Iterable[int]) -> list[Keyframe]:
        ids = list(ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM keyframes WHERE keyframe_id IN ({placeholders})", ids
        ).fetchall()
        by_id = {r["keyframe_id"]: _row_to_keyframe(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def get_keyframes_by_video(
        self, video_id: str, start_ms: int | None = None, end_ms: int | None = None,
    ) -> list[Keyframe]:
        query = "SELECT * FROM keyframes WHERE video_id = ?"
        params: list = [video_id]
        if start_ms is not None:
            query += " AND timestamp_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            query += " AND timestamp_ms <= ?"
            params.append(end_ms)
        query += " ORDER BY timestamp_ms"
        rows = self.conn.execute(query, params).fetchall()
        return [_row_to_keyframe(r) for r in rows]

    def get_keyframes_by_segment(self, segment_id: int) -> list[Keyframe]:
        rows = self.conn.execute(
            "SELECT * FROM keyframes WHERE scene_id = ? ORDER BY frame_id", (segment_id,)
        ).fetchall()
        return [_row_to_keyframe(r) for r in rows]

    def update_keyframe(
        self, keyframe_id: int,
        frame_idx: Optional[int] = None, timestamp_ms: Optional[int] = None,
        image_path: Optional[str] = None, scene_id: Optional[int] = None,
    ) -> Optional[Keyframe]:
        existing = self.get_keyframe(keyframe_id)
        if existing is None:
            return None
        frame_idx = existing.frame_idx if frame_idx is None else frame_idx
        timestamp_ms = existing.timestamp_ms if timestamp_ms is None else timestamp_ms
        image_path = existing.image_path if image_path is None else image_path
        scene_id = existing.scene_id if scene_id is None else scene_id
        self.conn.execute(
            "UPDATE keyframes SET frame_idx = ?, timestamp_ms = ?, image_path = ?, scene_id = ? WHERE keyframe_id = ?",
            (frame_idx, timestamp_ms, image_path, scene_id, keyframe_id),
        )
        self.conn.commit()
        return Keyframe(keyframe_id, existing.video_id, frame_idx, timestamp_ms, image_path, scene_id)

    def delete_keyframe(self, keyframe_id: int) -> bool:
        cur = self.conn.execute(
            "DELETE FROM keyframes WHERE keyframe_id = ?", (keyframe_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0



def _row_to_video(r: sqlite3.Row) -> Video:
    return Video(
        video_id=r["video_id"], video_path=r["video_path"],
        duration_ms=r["duration_ms"], fps=r["fps"],
    )
 
 
def _row_to_segment(r: sqlite3.Row) -> Segment:
    return Segment(
        segment_id=r["segment_id"], video_id=r["video_id"],
        frame_start=r["frame_start"], frame_end=r["frame_end"],
    )
 
 
def _row_to_keyframe(r: sqlite3.Row) -> Keyframe:
    return Keyframe(
        keyframe_id=r["keyframe_id"], video_id=r["video_id"],
        frame_idx=r["frame_idx"], timestamp_ms=r["timestamp_ms"],
        image_path=r["image_path"], scene_id=r["scene_id"],
    )

if __name__ == "__main__":
    db = MetadataDatabase("data/metadata.db")
    db.init_schema()