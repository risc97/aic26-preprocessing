# [HCMC AIC 2026] Data preprocessing

The preprocessing pipeline used by to process video data. The pipeline cuts video data into *keyframes*, then uses them to produce semantic embeddings (from multiple embedding models) and various other additional data types.

This pipeline is a component of a larger retrieval system, and its products would be directly used by the backend: [[HCMC AIC 2026] Backend](https://github.com/risc97/aic26-backend).

## Requirements

- A Python enviroment, preferrably Python 3.10 or above. Using virtual environment is recommended.
- [FFmpeg](https://www.ffmpeg.org/)
- [TransNetV2 package](https://github.com/YangTuanAnh/transnetv2_pytorch)
  - A script can be written to manually run TransNetV2 instead but using the package is recommended for convenience.
- Everything within `requirements.txt`.
  - Make sure all of them are installed by running `pip install -r requirements.txt`.

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
└── checkpoints/       # A custom path could be passed into the script instead
aic26-preprocessing/
```

Now the data can be processed by following the section below. After running the pipeline you should see additional directories storing the products of the process:

```
data/
├── videos/            # Videos
├── media-info/        # Metadata
├── staging/           # Keyframe timestamps in text files (VID_001.scenes.txt)
├── keyframes/         # Keyframes, organized by folders
│   ├── VID_001/
│   │   ├── 0001.jpeg
│   │   └── ...
│   └── ...
├── embeddings/        # Embedding shards
├── index/             # Embedding indices
├── checkpoints/       # Checkpoints
└── metadata.db
aic26-preprocessing/
```

## Running the pipeline

First start by setting up the environment:

- **NixOS**: run `nix-shell`. It automatically sets up a virtual environment and installs all requirements.
- **Other**: Make sure all dependencies in the [requirements](#requirements) are installed, then run `pip install -r requirements.txt` in your Python environment.

Should you want to use the competition's data instead of your own, download them using `./download_data.sh`. Note that the links might expire and the data could no longer be available. Don't forget to [prepare the data](#data-preparation) first.

Now move into this repository's directory and run these commands in order:
- `transnetv2_pytorch ../data/videos/ -o ../data/staging/` - runs keyframe detection
- `python init_db.py` - starts database to store metadata
- `python extract_keyframes.py` - extracts keyframes from detection results
- `python embed_keyframes.py` - embeds the extracted keyframes
- `python build_index.py` - builds indices for the embeddings

Each of the above scripts have their own arguments, which can be custom-passed and can be shown by running `<script-name> --help`. Specifically, it is highly recommended to view the arguments of `embed_keyframes.py` to see how one can pick a preferred embedding model to use.

## What's next?

The project is structured in such a manner that allows the user to extend their use to other models outside of the project's scope. One could add their own model by inheriting the base encoder in `models/base.py` and implement the model's class.

Once the indices building is done, see [[HCMC AIC 2026] Backend](https://github.com/risc97/aic26-backend) on how to use them.