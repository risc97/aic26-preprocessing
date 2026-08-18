from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data"

DB_PATH = DATA_PATH / "metadata.db"
SHARD_DIR = DATA_PATH / "embeddings"
CKPT_PATH = DATA_PATH / "checkpoints" / "c2lip.pt"
VIDEOS_DIR = DATA_PATH / "videos"
SCENES_DIR = DATA_PATH / "staging"
AUDIOS_DIR = DATA_PATH / "audios"
TRANSCRIPTS_DIR = DATA_PATH / "transcripts"
KEYFRAMES_DIR = DATA_PATH / "keyframes"
OCR_DIR = DATA_PATH / "ocr"
MEDIA_INFO_DIR = DATA_PATH / "media-info"
INDEX_PATH = DATA_PATH / "index"

KEYFRAME_MODE = "mid"


def stored_path(path: str | Path) -> str:
    """Convert a path to be relative to DATA_PATH.
    For example:
    DATA_PATH = Path('/workspace/data')
    stored_path(/workspace/data/keyframes/001.jpg) = keyframes/001.jpg
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(DATA_PATH))
    except ValueError:
        return str(resolved)


def resolve_stored_path(stored: str | Path) -> Path:
    """Restore a stored path to an absolute path."""
    return (DATA_PATH / stored).resolve()