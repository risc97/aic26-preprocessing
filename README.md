# [HCMC AIC 2026] Data preprocessing

Offline preprocessing for the TRISC video search engine at the HCMC AI Challenge 2026

## Overview

The system is split into three repositories:

| Repository | Role |
| --- | --- |
| **[aic26-preprocessing](https://github.com/risc97/aic26-preprocessing)** | Process raw videos into keyframes, embeddings, OCR, transcripts and search indices |
| [aic26-backend](https://github.com/risc97/aic26-backend) | FastAPI server to searches indices and serves keyframes and videos |
| [aic26-frontend](https://github.com/risc97/aic26-frontend2) | UI web app for searching, reviewing and submitting results |


Organize these repositories like this structure:

```text
aic26/
├── data/                  # input videos + preprocessing output
├── aic26-preprocessing/
├── aic26-backend/
└── aic26-frontend/
```

## Project structure

```text
aic26-preprocessing/
├── models/                       # Encoders
├── pipeline/                     # Shared logic used by the scripts below
├── download_data.sh              # Script to download data
├── find_ad_span.py               # Find ad segments from a reference clip
├── exclude_spans.py              # Drop those segments from the scene lists
├── split_slides.py               # Split lecture shots at slide changes
├── resample_traffic_camera.py    # Subdivide static traffic-camera shots
├── extract_keyframes.py          # Read scene lists to extract keyframes
├── embed_keyframes.py            # Embed keyframes into embedding shards
├── build_index.py                # Index embedding shards into vector index
├── convert_to_audio.sh           # Convert videos to audio tracks
├── extract_transcript.py         # Use Gipformer to extract transcripts from audio
├── embed_transcript.py           # Embed transcripts into embeddings
├── ocr_keyframes.py              # Extract OCR text from keyframes
└── detect_keyframes.py           # Extract OWLv2 embeddings
```
## Usage

### Prerequisites
- **Python** 3.10+, tested on 3.12
- **FFmpeg** and **GNU parallel**
- **CUDA GPU** (recommended)

### Installation
```bash
pwd # Make sure you are in aic26/
git clone https://github.com/risc97/aic26-preprocessing.git
cd aic26-preprocessing
```

- **NixOS:** run `nix-shell`
- **Other:**
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# TransNetV2
git clone --depth 1 https://github.com/YangTuanAnh/transnetv2_pytorch.git .venv/vendor/transnetv2_pytorch
pip install -e .venv/vendor/transnetv2_pytorch
```

PaddlePaddle is only needed for OCR (`ocr_keyframes.py`). It is optional, and the install depends on your CUDA version. See the Paddle block in [shell.nix](./shell.nix) for the exact steps.

## Data preparation

The video data should be organized as such:

```
data/
├── videos/            # Videos, in this exact directory
│   ├── VID_001.mp4
│   ├── VID_097.webm
│   └── ...
├── media-info/        # JSON metadata files, should match the clip's name
│   ├── VID_001.json
│   ├── VID_097.json
│   └── ...
└── checkpoints/       # Your fine-tuned checkpoint
```

## Run

```bash
# 1. Shot detection
transnetv2_pytorch ../data/videos/ -o ../data/staging/

# 2. Content-aware refinement
## Post-preprocess recorded lesson videos
python find_ad_span.py --ref-video L25_V060 --ref-start 0 --ref-end 67
python find_ad_span.py --ref-video L25_V060 --ref-start 1388 --ref-end 2886
python find_ad_span.py --ref-video L25_V060 --ref-start 18790 --ref-end 20898
python find_ad_span.py --ref-video L25_V060 --ref-start 37957 --ref-end 39455
python exclude_spans.py

python split_slides.py

## Post-preprocess static traffic camera
python resample_traffic_camera.py

# 3. Keyframes
python extract_keyframes.py

# 4. Text-image embeddings (model: siglip, siglip2, pe)
python embed_keyframes.py --model siglip2 --batch-size 32
python build_index.py --model siglip2

# 5. Visual embeddings
python embed_keyframes.py --model dinov3 --batch-size 96
python build_index.py --model dinov3

# 6. Speech transcripts
./convert_to_audio.sh
python extract_transcript.py --audio-dir data/audios
python embed_transcript.py
python build_index.py --model gte --index data/index/gte-transcripts.tvim

# 7. OCR and object detection
python ocr_keyframes.py
python detect_keyframes.py
```
