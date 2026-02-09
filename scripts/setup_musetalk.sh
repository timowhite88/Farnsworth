#!/usr/bin/env bash
#
# Setup MuseTalk on the Farnsworth server (RunPod V100).
# Run this once to clone the repo, install deps, and download weights.
#
# Usage:  bash scripts/setup_musetalk.sh
#

set -e

MUSETALK_DIR="/workspace/MuseTalk"
FARNSWORTH_DIR="/workspace/Farnsworth"

echo "================================================"
echo "  MuseTalk Setup for Farnsworth VTuber"
echo "================================================"

# 1. Clone MuseTalk
if [ ! -d "$MUSETALK_DIR" ]; then
    echo "[1/5] Cloning MuseTalk..."
    cd /workspace
    git clone https://github.com/TMElyralab/MuseTalk.git
else
    echo "[1/5] MuseTalk already cloned at $MUSETALK_DIR"
fi

cd "$MUSETALK_DIR"

# 2. Install Python dependencies
echo "[2/5] Installing Python dependencies..."
pip install -q diffusers==0.30.2 accelerate==0.28.0 transformers==4.39.2
pip install -q soundfile librosa einops omegaconf ffmpeg-python moviepy
pip install -q opencv-python-headless gdown requests imageio[ffmpeg]

# Install mmlab packages (for DWPose face detection)
pip install -q --no-cache-dir -U openmim
mim install -q mmengine 2>/dev/null || true
mim install -q "mmcv>=2.0.1" 2>/dev/null || true
mim install -q "mmdet>=3.1.0" 2>/dev/null || true
mim install -q "mmpose>=1.1.0" 2>/dev/null || true

# Install local whisper package
if [ -d "$MUSETALK_DIR/musetalk/whisper" ]; then
    pip install -q --editable "$MUSETALK_DIR/musetalk/whisper"
fi

# Install face_alignment
pip install -q face_alignment

# 3. Download model weights
echo "[3/5] Downloading model weights (~6.8 GB total)..."
mkdir -p models/musetalkV15 models/sd-vae-ft-mse models/whisper models/dwpose models/face-parse-bisent

# MuseTalk v1.5 weights
if [ ! -f "models/musetalkV15/unet.pth" ]; then
    echo "  Downloading MuseTalk v1.5 UNet..."
    huggingface-cli download TMElyralab/MuseTalk musetalkV15/musetalk.json --local-dir ./models --quiet
    huggingface-cli download TMElyralab/MuseTalk musetalkV15/unet.pth --local-dir ./models --quiet
else
    echo "  MuseTalk v1.5 weights already present"
fi

# SD-VAE
if [ ! -f "models/sd-vae-ft-mse/diffusion_pytorch_model.bin" ]; then
    echo "  Downloading SD-VAE..."
    huggingface-cli download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.bin --local-dir ./models/sd-vae-ft-mse --quiet
else
    echo "  SD-VAE weights already present"
fi

# Whisper-tiny
if [ ! -f "models/whisper/pytorch_model.bin" ]; then
    echo "  Downloading Whisper-tiny..."
    huggingface-cli download openai/whisper-tiny --local-dir ./models/whisper --quiet
else
    echo "  Whisper weights already present"
fi

# DWPose
if [ ! -f "models/dwpose/dw-ll_ucoco_384.pth" ]; then
    echo "  Downloading DWPose..."
    huggingface-cli download yzd-v/DWPose dw-ll_ucoco_384.pth --local-dir ./models/dwpose --quiet
else
    echo "  DWPose weights already present"
fi

# Face parser - ResNet18
if [ ! -f "models/face-parse-bisent/resnet18-5c106cde.pth" ]; then
    echo "  Downloading ResNet18 backbone..."
    wget -q https://download.pytorch.org/models/resnet18-5c106cde.pth -O models/face-parse-bisent/resnet18-5c106cde.pth
else
    echo "  ResNet18 backbone already present"
fi

# Face parser - 79999_iter.pth
if [ ! -f "models/face-parse-bisent/79999_iter.pth" ]; then
    echo "  Downloading face parser model..."
    gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth --quiet
else
    echo "  Face parser weights already present"
fi

# 4. Verify FFmpeg
echo "[4/5] Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "  FFmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "  FFmpeg not found! Installing..."
    apt-get update -qq && apt-get install -y -qq ffmpeg
fi

# 5. Quick import test
echo "[5/5] Verifying installation..."
cd "$FARNSWORTH_DIR"
python -c "
import sys
sys.path.insert(0, '$MUSETALK_DIR')
from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.audio_processor import AudioProcessor
print('MuseTalk imports OK')
"

echo ""
echo "================================================"
echo "  MuseTalk Setup Complete!"
echo "  Dir: $MUSETALK_DIR"
echo "  Weights: $MUSETALK_DIR/models/"
echo ""
echo "  Start streaming with:"
echo "    python scripts/start_musetalk_stream.py"
echo "================================================"
