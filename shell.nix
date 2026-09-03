{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-25.05.tar.gz")
    { config.allowUnfree = true; } }:

# Pinned to nixos-25.05 for python3.10

let
  dynamic-libs = with pkgs; [
    stdenv.cc.cc.lib # Provides libstdc++.so.6 (crucial for TensorFlow/Keras/Numpy)
    zlib             # Used for data compression
    glib             # Common dependency for underlying C libraries
    freetype         # Needed by wordcloud to render fonts
    libjpeg          # Needed by image processing libraries
    libGL            # Needed by opencv-python and mediapipe
    libsndfile       # Needed by librosa (via soundfile)
  ];
in
pkgs.mkShell {
  packages = with pkgs; [
    python312
    gcc
    gnumake
    ngrok
    wget
    curl
    ffmpeg
    parallel
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath dynamic-libs}:$LD_LIBRARY_PATH"

    if [ -d /run/opengl-driver/lib ]; then
      export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
    else
      echo "warning: /run/opengl-driver/lib not found -- GPU may be unavailable"
    fi

    if [ ! -d ".venv" ]; then
      echo "Creating venv..."
      python3.12 -m venv .venv
    fi

    # Activate the environment
    source .venv/bin/activate

    # Pin the constraints for paddlepaddle_gpu
    CONSTRAINTS="$PWD/.venv/constraints-cuda.txt"
    {
      echo "torch==2.13.0"
      # torch 2.13.0's CUDA stack
      echo "nvidia-cublas==13.1.1.3"
      echo "nvidia-cuda-runtime==13.0.96"
      echo "nvidia-cudnn-cu13==9.20.0.48"
      echo "nvidia-nccl-cu13==2.29.7"
      echo "nvidia-nvjitlink==13.3.33"
      echo "nvidia-cuda-nvrtc==13.0.88"

      echo "cuda-python==13.0.3"
      echo "nvidia-cuda-cccl==13.0.85"
      echo "opt-einsum==3.3.0"

      echo "pyyaml==6.0.2"

      echo "setuptools<82"
    } > "$CONSTRAINTS"

    export PIP_CONSTRAINT="$CONSTRAINTS"

    # Install the requirements
    if [ -f "requirements.txt" ] && [ ! -f ".venv/.requirements-installed" ]; then
      echo "Downloading requirements.txt..."
      # Using --prefer-binary prevents pip from trying to compile heavy ML packages from source
      pip install --prefer-binary -r requirements.txt && touch .venv/.requirements-installed
    fi

    PADDLE_WHEEL="paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl"
    if [ ! -f ".venv/.paddle-installed" ] && [ "$(uname -m)" = "x86_64" ]; then
      echo "Installing paddlepaddle-gpu 3.3.1..."
      mkdir -p .venv/cache
      curl -L -C - -o ".venv/cache/$PADDLE_WHEEL" \
        "https://paddle-whl.cdn.bcebos.com/stable/cu130/paddlepaddle-gpu/$PADDLE_WHEEL"

      pip install --no-deps ".venv/cache/$PADDLE_WHEEL"

      pip install --no-deps paddleocr paddlex

      pip install cuda-python nvidia-cuda-cccl opt-einsum \
        aistudio-sdk chardet colorlog modelscope prettytable py-cpuinfo \
        ruamel-yaml ujson aiohttp
      pip install vietocr
      pip uninstall -y opencv-python opencv-python-headless
      pip install opencv-python-headless

      touch .venv/.paddle-installed
      echo "Paddle installed"
    fi

    # TransNetV2 PyTorch
    VENDOR_DIR=".venv/vendor"
    if [ ! -d "$VENDOR_DIR/transnetv2_pytorch" ]; then
      echo "Cloning transnetv2_pytorch..."
      mkdir -p "$VENDOR_DIR"
      git clone --depth 1 https://github.com/YangTuanAnh/transnetv2_pytorch.git "$VENDOR_DIR/transnetv2_pytorch"
    fi

    if [ ! -f ".venv/.transnetv2-installed" ]; then
      echo "Installing transnetv2_pytorch (editable)..."
      pip install --prefer-binary ffmpeg-python torch pillow
      pip install -e "$VENDOR_DIR/transnetv2_pytorch" && touch .venv/.transnetv2-installed
    fi

    echo "Environment done"
  '';
}
