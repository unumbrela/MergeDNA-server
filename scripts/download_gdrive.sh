#!/bin/bash
# Download datasets from Google Drive that require manual download
# Run this script from the MergeDNA root directory
#
# If gdown fails due to rate limiting, download manually from browser:
#   GUE:     https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view
#   Pretrain: https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  MergeDNA Dataset Download (Google Drive)"
echo "============================================"

# ---- GUE Benchmark (24 tasks, ~298MB) ----
GUE_DIR="$ROOT_DIR/data/gue_benchmark"
mkdir -p "$GUE_DIR"

if [ ! -d "$GUE_DIR/GUE" ]; then
    echo ""
    echo "[1/2] Downloading GUE Benchmark (~298MB)..."
    cd "$GUE_DIR"

    if command -v gdown &> /dev/null; then
        gdown "1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-" -O GUE.zip || {
            echo "ERROR: gdown failed. Please download manually:"
            echo "  URL: https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view"
            echo "  Save to: $GUE_DIR/GUE.zip"
            echo "  Then run: cd $GUE_DIR && unzip GUE.zip"
        }
    else
        echo "gdown not installed. Install: pip install gdown"
        echo "Or download manually from:"
        echo "  https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view"
    fi

    if [ -f "$GUE_DIR/GUE.zip" ]; then
        echo "Extracting GUE.zip..."
        unzip -o GUE.zip
        echo "GUE Benchmark extracted successfully!"
    fi
else
    echo "[1/2] GUE Benchmark already exists, skipping."
fi

# ---- Multi-Species Genomes (pretrain, ~9.3GB) ----
PRETRAIN_DIR="$ROOT_DIR/data/pretrain"
mkdir -p "$PRETRAIN_DIR"

if [ ! -d "$PRETRAIN_DIR/multi_species_genomes" ] && [ ! -f "$PRETRAIN_DIR/multi_species_genomes.zip" ]; then
    echo ""
    echo "[2/2] Downloading Multi-Species Genomes (~9.3GB)..."
    echo "This may take a long time. Consider downloading via browser."
    cd "$PRETRAIN_DIR"

    if command -v gdown &> /dev/null; then
        gdown "1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f" -O multi_species_genomes.zip || {
            echo "ERROR: gdown failed. Please download manually:"
            echo "  URL: https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view"
            echo "  Save to: $PRETRAIN_DIR/multi_species_genomes.zip"
            echo "  Then run: cd $PRETRAIN_DIR && unzip multi_species_genomes.zip"
        }
    else
        echo "gdown not installed. Or download manually from:"
        echo "  https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view"
    fi

    if [ -f "$PRETRAIN_DIR/multi_species_genomes.zip" ]; then
        echo "Extracting multi_species_genomes.zip..."
        unzip -o multi_species_genomes.zip
        echo "Multi-Species Genomes extracted successfully!"
    fi
else
    echo "[2/2] Multi-Species Genomes already exists, skipping."
fi

echo ""
echo "============================================"
echo "  Download Summary"
echo "============================================"
echo ""
echo "Data directory: $ROOT_DIR/data/"
ls -la "$ROOT_DIR/data/"
echo ""
echo "If any downloads failed, use browser to download from Google Drive:"
echo "  GUE:      https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view"
echo "  Pretrain:  https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view"
