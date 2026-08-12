from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data"

DB_PATH = DATA_PATH / "metadata.db"
SHARD_DIR = DATA_PATH / "embeddings"
CKPT_PATH = DATA_PATH / "checkpoints" / "c2lip.pt"
VIDEOS_DIR = DATA_PATH / "videos"
SCENES_DIR = DATA_PATH / "staging"
KEYFRAMES_DIR = DATA_PATH / "keyframes"
MEDIA_INFO_DIR = DATA_PATH / "media-info"
INDEX_PATH = DATA_PATH / "index" / "keyframes.tvim"