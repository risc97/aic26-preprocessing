# AIC26 Video Search

## Overview
This project include multiple repositories:
- [aic26-preprcessing](https://github.com/risc97/aic26-preprocessing): Turn raw videos into keyframes, embeddings, and a search index
- [aic26-backend](https://github.com/risc97/aic26-backend): FastAPI server that searches the index and serves keyframes and videos

## Project structure

You may organize the project structure like this:
```
aic26/
├── data/
│   ├── videos/            # L21_V001.webm ...
│   ├── media-info/        # L21_V001.json ...
│   ├── staging/           # L21_V001.scenes.txt ... (from TransNetV2)
│   ├── keyframes/         # L21_V001/0001.jpg ...
│   ├── embeddings/        # embedding shards
│   ├── index/             # embedding index
│   ├── checkpoints/       # c2lip.pt
│   └── metadata.db
├── aic26-preprocessing/
└── aic26-backend/
```

## Install

```bash
mkdir aic26 && cd aic26
git clone https://github.com/risc97/aic26-preprocessing.git
git clone https://github.com/risc97/aic26-backend.git
cd aic26-preprocessing
```

Set up the environment:

- **NixOS**: run `nix-shell`. It creates `.venv`, installs `requirements.txt`, and installs TransNetV2.
- **Other**: make a Python 3.10 venv, `pip install -r requirements.txt`, install `ffmpeg`, and install [transnetv2_pytorch](https://github.com/YangTuanAnh/transnetv2_pytorch).

## Build the data

Run these from `aic26-preprocessing/`, in order:

```bash
# download the data from Organizer's source (maybe unavailable)
./download_data.sh

# find scene boundary
transnetv2_pytorch ../data/videos/ -o ../data/staging/
# create metadata.db
python init_db.py
# extract keyframe and save infomation to db
python extract_keyframes.py

# embed keyframes -> data/embeddings/
python embed_keyframes.py --model c2lip
python embed_keyframes.py --model siglip2 --batch-size 32
# build data/index/keyframes.tvim
python build_index.py
```

Some utils command that you might need:

```bash
# clean the C2LIP weight -> checkpoints/c2lip.pt
python strip_checkpoint.py

```

## Run the backend

Once the index exists, follow the [aic26-backend](https://github.com/risc97/aic26-backend) README. It serves `POST /query` for search, plus endpoints for keyframe images and videos.