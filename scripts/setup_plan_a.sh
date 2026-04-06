#!/bin/bash
# MergeDNA-Long (Plan A) 环境准备脚本
# 用法: bash scripts/setup_plan_a.sh

set -e

echo "=========================================="
echo "MergeDNA-Long: Plan A Setup"
echo "=========================================="

# ========== 1. Python依赖 ==========
echo ""
echo "[1/5] Installing Python dependencies..."

pip install flash-linear-attention 2>/dev/null || echo "WARN: flash-linear-attention install failed, will try from source later"
pip install pyfaidx 2>/dev/null || echo "WARN: pyfaidx install failed"
pip install pandas scipy 2>/dev/null

# 验证关键库
python -c "import mamba_ssm; print(f'  mamba_ssm: {mamba_ssm.__version__}')" 2>/dev/null || echo "  WARN: mamba_ssm not available"
python -c "import flash_attn; print(f'  flash_attn: {flash_attn.__version__}')" 2>/dev/null || echo "  WARN: flash_attn not available"
python -c "import fla; print('  flash-linear-attention: OK')" 2>/dev/null || echo "  WARN: flash-linear-attention not available"
python -c "import triton; print(f'  triton: {triton.__version__}')" 2>/dev/null || echo "  WARN: triton not available"

# ========== 2. 论文目录 ==========
echo ""
echo "[2/5] Setting up papers directory..."

mkdir -p papers/plan_a

cat > papers/plan_a/DOWNLOAD_LIST.md << 'PAPEREOF'
# Plan A 参考论文下载清单

请手动下载以下论文PDF到当前目录：

1. **BLT: Byte Latent Transformer**
   - https://arxiv.org/abs/2412.09871
   - 重点阅读: Section 3.1 (Entropy-based patching)

2. **Gated DeltaNet: Improving Linear Transformers with Gated Delta Rule**
   - https://arxiv.org/abs/2412.06464
   - 重点阅读: Section 3 (Architecture), Section 4 (Experiments)

3. **HybriDNA: A Hybrid Transformer-Mamba2 Long-Range DNA Language Model**
   - https://arxiv.org/abs/2502.10807
   - 重点阅读: Section 3 (Hybrid architecture design), Table 1 (6:1 ratio finding)

4. **DTEM: Learning to Merge Tokens via Decoupled Embedding**
   - https://arxiv.org/abs/2410.13228
   - 重点阅读: Section 3 (Decoupled merge embedding)

5. **DiffRate: Differentiable Compression Rate for Token Merging**
   - https://arxiv.org/abs/2305.17997
   - 重点阅读: Section 3 (Learnable per-layer rate)

6. **Mixture-of-Depths: Dynamically Allocating Compute**
   - https://arxiv.org/abs/2404.02258
   - 参考: token-level routing思想

7. **Mamba-2: State Space Duality**
   - https://arxiv.org/abs/2405.21060
   - 参考: SSM基础方法

8. **MaMe: Token Reduction with Mamba Models**
   - https://arxiv.org/abs/2508.13599
   - 参考: Token Merging + SSM的结合

PAPEREOF

echo "  Papers download list created at papers/plan_a/DOWNLOAD_LIST.md"

# ========== 3. 参考仓库 ==========
echo ""
echo "[3/5] Cloning additional reference repos..."

cd reference_repos/

if [ ! -d "fla" ]; then
    echo "  Cloning flash-linear-attention..."
    git clone --depth 1 https://github.com/fla-org/flash-linear-attention fla 2>/dev/null || echo "  WARN: Failed to clone fla"
else
    echo "  flash-linear-attention already exists, skipping"
fi

if [ ! -d "diffrate" ]; then
    echo "  Cloning DiffRate..."
    git clone --depth 1 https://github.com/OpenMMLab/DiffRate diffrate 2>/dev/null || echo "  WARN: Failed to clone DiffRate"
else
    echo "  DiffRate already exists, skipping"
fi

cd ..

# ========== 4. LRB数据 ==========
echo ""
echo "[4/5] Setting up LRB benchmark data..."

mkdir -p ./data/external_refs
if [ ! -d "./data/external_refs/genomics-long-range-benchmark" ]; then
    echo "  Cloning LRB benchmark repo..."
    git clone --depth 1 https://github.com/Trop-LongRange/genomics-long-range-benchmark \
        ./data/external_refs/genomics-long-range-benchmark 2>/dev/null || \
        echo "  WARN: Failed to clone LRB repo"
else
    echo "  LRB benchmark repo already exists, skipping"
fi

mkdir -p ./data/lrb/raw
echo "  NOTE: You need to manually download reference genomes:"
echo "    wget -O ./data/lrb/raw/hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
echo "    wget -O ./data/lrb/raw/hg19.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
echo "    gunzip ./data/lrb/raw/hg38.fa.gz ./data/lrb/raw/hg19.fa.gz"

# ========== 5. 目录结构 ==========
echo ""
echo "[5/5] Creating output directories..."

mkdir -p outputs/pretrain_long
mkdir -p outputs/finetune_long
mkdir -p outputs/ablation
mkdir -p outputs/lrb_long
mkdir -p outputs/analysis

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download papers to papers/plan_a/"
echo "  2. Download reference genomes (hg38/hg19)"
echo "  3. Download pretrain data: python scripts/download_data.py"
echo "  4. Implement code modules (see docs/PLAN_A_IMPLEMENTATION_GUIDE.md)"
echo "  5. Run local test: python train.py --config configs/pretrain_long_local.yaml --mode pretrain --max_steps 50"
echo "=========================================="
