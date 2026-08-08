#!/usr/bin/env bash

TARGET_DIR="data/video"

DELETE_ZIP="${DELETE_ZIP:-0}"

mkdir -p "$TARGET_DIR" data/keyframes data/staging data/embeddings

URLS=(
    "https://aic-data.ledo.io.vn/Videos_L21_a.zip"
    # "https://aic-data.ledo.io.vn/Videos_L22_a.zip"
    # "https://aic-data.ledo.io.vn/Videos_L23_a.zip"
    # "https://aic-data.ledo.io.vn/Videos_L24_a.zip"
    # "https://aic-data.ledo.io.vn/Videos_L25_a.zip"
    # "https://aic-data.ledo.io.vn/Videos_L26_a.zip"
    # "https://aic-data.ledo.io.vn/Videos_L26_b.zip"
    # "https://aic-data.ledo.io.vn/Videos_L26_c.zip"
    # "https://aic-data.ledo.io.vn/Videos_L26_d.zip"
    # "https://aic-data.ledo.io.vn/Videos_L26_e.zip"
    # "https://aic-data.ledo.io.vn/Videos_L27_a.zip"
    # "https://aic-data.ledo.io.vn/Videos_L28_a.zip"
    # "https://aic-data.ledo.io.vn/Videos_L29_a.zip"
    # "https://aic-data.ledo.io.vn/Videos_L30_a.zip"
)

for url in "${URLS[@]}"; do
    zip_name="$(basename "$url")"
    zip_path="$TARGET_DIR/$zip_name"

    echo "==> Downloading $zip_name"
    wget -t 10 -c -P "$TARGET_DIR" "$url"

    echo "==> Extracting $zip_name"
    unzip -n -j -q "$zip_path" -d "$TARGET_DIR"

    echo "==> Removing $zip_name"
    rm -f "$zip_path"
done

wget -t 10 -c -P "data/" "https://aic-data.ledo.io.vn/media-info-aic25-b1.zip"
unzip data/media-info-aic25-b1.zip -d data/
rm -f data/media-info-aic25-b1.zip

echo "==> Done"
