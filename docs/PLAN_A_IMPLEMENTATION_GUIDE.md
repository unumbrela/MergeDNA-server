# MergeDNA-Long 实施指南

> **"MergeDNA-Long: Scaling Context-Aware Genome Modeling to 100K Bases via Hybrid Token Merging"**
>
> 组合：Entropy-Guided Merging + Hybrid SSM-Attention Latent Encoder + Learned Compression Schedule

---

## 一、总体目标

将MergeDNA从4K上下文扩展到32K-100K，同时提升token merging质量，在长程基因组基准（LRB）上达到SOTA，并在标准基准（GB/NT/GUE）上保持或超越原始性能。

### 三个核心改进
1. **Entropy-Guided Token Merging**：在Local Encoder的合并决策中引入局部熵信号，让高信息区（调控元件、splice sites）保留细粒度
2. **Hybrid SSM-Attention Latent Encoder**：将20层全注意力替换为17层Gated DeltaNet + 3层Full Attention，O(N)复杂度
3. **Learned Compression Schedule**：每层的合并数r不再预设，通过Gumbel-Softmax端到端学习

---

## 二、需要下载的资源

### 2.1 论文（存放到 `papers/plan_a/`）

| # | 论文 | ArXiv/链接 | 用途 |
|---|------|----------|------|
| 1 | **BLT: Byte Latent Transformer** | arxiv.org/abs/2412.09871 | Entropy-based patching核心方法 |
| 2 | **Gated DeltaNet** | arxiv.org/abs/2412.06464 | 最强线性注意力替代方案 |
| 3 | **HybriDNA** | arxiv.org/abs/2502.10807 | 混合SSM-Attention在DNA上的验证 |
| 4 | **DTEM** (Lee & Hong, NeurIPS 2024) | arxiv.org/abs/2410.13228 | 解耦merge embedding |
| 5 | **DiffRate** | arxiv.org/abs/2305.17997 | 可微分压缩率调度 |
| 6 | **Mixture-of-Depths** | arxiv.org/abs/2404.02258 | 可选：自适应计算深度参考 |
| 7 | **MaMe: Token Merging for SSMs** | arxiv.org/abs/2508.13599 | Token Merging + SSM的结合 |
| 8 | **NTv3** | biorxiv.org/content/10.64898/2025.12.22.695963v1 | U-Net压缩+Transformer先例 |
| 9 | **Mamba-2: State Space Duality** | arxiv.org/abs/2405.21060 | SSM基础方法 |

### 2.2 代码库

#### 已有的参考仓库（`reference_repos/` 中）
- `reference_repos/blt/` — BLT参考实现（entropy patching）
- `reference_repos/hnet/` — HNet可微分分割
- `reference_repos/Mixture-of-depths/` — MoD参考
- `reference_repos/mamba/` — Mamba SSM参考

#### 可选：克隆额外仓库到 `reference_repos/`
```bash
cd reference_repos/

# flash-linear-attention（查看DeltaNet实现细节）
git clone https://github.com/fla-org/flash-linear-attention fla

# DiffRate（可微分压缩率调度参考）
git clone https://github.com/OpenGVLab/DiffRate diffrate
```

### 2.3 数据

#### 预训练数据（已配置）
- Multi-Species Genomes：`./data/pretrain/multi_species_genomes/`
- 来源：HuggingFace `InstaDeepAI/multi_species_genomes`（DNABERT-2扩展版）

#### 评估数据

| 数据集 | 路径 | 获取方式 |
|--------|------|---------|
| Genomic Benchmark (8任务) | `./data/genomic_benchmark_sequences/` | `python scripts/download_data.py --dataset genomic_benchmark` |
| NT Benchmark (18任务) | `./data/nt_benchmark/` | HuggingFace自动下载 |
| GUE Benchmark (24任务) | `./data/gue_benchmark/GUE/` | `python scripts/download_data.py --dataset gue` |
| **LRB eQTL** (20Kbp) | `./data/lrb/` | 见下方LRB下载命令 |
| **LRB Bulk RNA** (40Kbp) | `./data/lrb/` | 见下方LRB下载命令 |
| hg38参考基因组 | `./data/lrb/raw/hg38.fa` | UCSC Genome Browser |
| hg19参考基因组 | `./data/lrb/raw/hg19.fa` | UCSC Genome Browser |

**LRB数据下载**：
```bash
# 1. 克隆LRB仓库
git clone https://github.com/Trop-LongRange/genomics-long-range-benchmark \
    ./data/external_refs/genomics-long-range-benchmark

# 2. 下载参考基因组
mkdir -p ./data/lrb/raw
wget -O ./data/lrb/raw/hg38.fa.gz \
    https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
wget -O ./data/lrb/raw/hg19.fa.gz \
    https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip ./data/lrb/raw/hg38.fa.gz
gunzip ./data/lrb/raw/hg19.fa.gz

# 3. 安装pyfaidx（LRB需要）
pip install pyfaidx
```

---

## 三、要实现的代码模块

### 概览：新增/修改文件清单

```
mergedna/
├── model/
│   ├── token_merging.py          # [修改] 添加EntropyGuidedTokenMerging
│   ├── entropy_model.py          # [新增] 轻量熵预测模块
│   ├── local_encoder.py          # [修改] 支持entropy-guided merging + learned schedule
│   ├── hybrid_layers.py          # [新增] Gated DeltaNet / Mamba wrapper
│   ├── latent_encoder.py         # [修改] 添加HybridLatentEncoder
│   ├── mergedna.py               # [修改] 添加MergeDNALongConfig
│   └── transformer.py            # [不变]
├── training/
│   ├── pretrain.py               # [修改] 支持新config
│   └── losses.py                 # [不变]
configs/
├── pretrain_long.yaml            # [新增] MergeDNA-Long预训练配置
├── pretrain_long_local.yaml      # [新增] 本地调试配置
├── finetune_long.yaml            # [新增] MergeDNA-Long微调配置
scripts/
├── benchmark_efficiency.py       # [新增] 效率对比脚本
├── run_plan_a.sh                 # [新增] 一键运行全流程
```

---

### 模块1：Entropy-Guided Token Merging

**文件**：`mergedna/model/entropy_model.py`（新增）

```python
"""轻量局部熵预测模块。

使用小型causal convolution网络估计每个碱基位置的局部信息熵。
高熵位置（信息丰富）在token merging中被保护，低熵位置被优先合并。
"""

class LocalEntropyEstimator(nn.Module):
    """轻量熵估计器（~1M参数）。
    
    架构：3层1D causal convolution + sigmoid输出
    输入：[B, N, D] token embeddings
    输出：[B, N] 每个位置的entropy score (0-1)
    """
    def __init__(self, embed_dim=1024, hidden_dim=128, kernel_size=9):
        ...
    def forward(self, x): -> [B, N] entropy scores
        ...
```

**文件**：`mergedna/model/token_merging.py`（修改）

在 `LocalWindowTokenMerging` 中集成entropy信号：

```python
class EntropyGuidedTokenMerging(LocalWindowTokenMerging):
    """Entropy-aware token merging。
    
    merge_score = similarity - alpha * entropy_penalty
    高熵token的entropy_penalty高 → 不容易被合并
    """
    def __init__(self, embed_dim, window_size=16, entropy_weight=0.5):
        super().__init__(embed_dim, window_size)
        self.entropy_weight = entropy_weight  # α

    def forward(self, x, source, r, attention_mask=None, entropy_scores=None):
        # 修改_bipartite_soft_matching，在scores中减去entropy_penalty
        ...
```

**关键设计**：
- entropy score通过BLT启发的方式计算：对embedding做轻量卷积预测下一个碱基的条件概率，取负对数期望
- α（entropy_weight）作为超参数，可在0.1-1.0之间调节
- 训练时entropy model与主模型端到端联合训练

---

### 模块2：Learned Compression Schedule

**文件**：`mergedna/model/local_encoder.py`（修改）

```python
class LearnedCompressionSchedule(nn.Module):
    """可学习的逐层压缩率调度。
    
    每层有一个可学习的logit，通过sigmoid映射到[r_min, r_max]。
    训练时用Gumbel-Softmax添加噪声，推理时用确定性值。
    """
    def __init__(self, num_layers, window_size, r_min=1, r_max=None):
        self.r_logits = nn.Parameter(torch.zeros(num_layers))
        self.r_min = r_min
        self.r_max = r_max or window_size // 2
        
    def forward(self, layer_idx, training=True):
        # sigmoid映射到[r_min, r_max]
        r_float = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(self.r_logits[layer_idx])
        if training:
            # Gumbel noise for exploration
            noise = torch.rand_like(r_float) * 0.5
            r_float = r_float + noise
        return r_float.round().int().clamp(self.r_min, self.r_max)
```

---

### 模块3：Hybrid SSM-Attention Latent Encoder

**文件**：`mergedna/model/hybrid_layers.py`（新增）

```python
"""Hybrid layers wrapping Gated DeltaNet and Mamba for the Latent Encoder.

提供统一接口，使得TransformerBlock和SSM Block可以互换使用。
"""

class GatedDeltaNetBlock(nn.Module):
    """Gated DeltaNet block wrapper。
    
    使用flash-linear-attention库的GatedDeltaNet实现。
    接口与TransformerBlock保持一致：
        forward(x, attention_mask) -> (x, None)
    """
    def __init__(self, embed_dim, num_heads=16, ...):
        from fla.layers import GatedDeltaNet
        self.norm1 = RMSNorm(embed_dim)
        self.ssm = GatedDeltaNet(d_model=embed_dim, ...)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim)
    
    def forward(self, x, attention_mask=None):
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x, None


class MambaBlock(nn.Module):
    """Mamba-2 block wrapper（备选方案）。"""
    def __init__(self, embed_dim, ...):
        from mamba_ssm import Mamba2
        ...


class HybridLatentEncoder(nn.Module):
    """混合SSM-Attention Latent Encoder。
    
    配置示例（20层，3层attention at 6/12/18）：
        layer_types = ["ssm"]*5 + ["attn"] + ["ssm"]*5 + ["attn"] + ["ssm"]*5 + ["attn"] + ["ssm"]*2
    
    参数量与原始20层全注意力相近（SSM层参数略少，但层数不变）。
    """
    def __init__(self, embed_dim, num_layers=20, num_heads=16,
                 ssm_type="gated_deltanet",      # "gated_deltanet" | "mamba2"
                 attention_layer_indices=[5,11,17],  # 哪些层用Full Attention
                 ...):
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i in attention_layer_indices:
                self.layers.append(TransformerBlock(...))
            else:
                if ssm_type == "gated_deltanet":
                    self.layers.append(GatedDeltaNetBlock(...))
                elif ssm_type == "mamba2":
                    self.layers.append(MambaBlock(...))
        self.norm = RMSNorm(embed_dim)
        self.global_tome = GlobalTokenMerging(embed_dim)
```

---

### 模块4：MergeDNA-Long Config

**文件**：`mergedna/model/mergedna.py`（修改）

```python
@dataclass
class MergeDNALongConfig(MergeDNAConfig):
    """MergeDNA-Long扩展配置。"""
    
    # Entropy-guided merging
    use_entropy_guided_merging: bool = True
    entropy_weight: float = 0.5        # α: entropy对merge score的权重
    entropy_model_hidden_dim: int = 128
    entropy_model_kernel_size: int = 9
    
    # Learned compression schedule
    use_learned_compression: bool = True
    r_min_per_window: int = 1
    r_max_per_window: int = 8          # window_size // 2
    
    # Hybrid Latent Encoder
    latent_encoder_type: str = "hybrid"  # "transformer" | "hybrid"
    ssm_type: str = "gated_deltanet"     # "gated_deltanet" | "mamba2"
    attention_layer_indices: list = field(default_factory=lambda: [5, 11, 17])
    
    # 扩展上下文
    max_seq_length: int = 32768        # 32K default, 可扩展到100K
    compression_target: float = 0.25   # 更激进的压缩（原始0.5）
```

---

## 四、配置文件

### 4.1 预训练配置 `configs/pretrain_long.yaml`

```yaml
# MergeDNA-Long Pre-training (A800 80GB)
# 核心改进：entropy-guided merging + hybrid latent encoder + learned compression

# Data
data_path: "./data/pretrain/multi_species_genomes"
max_seq_length: 32768            # 32K (原始: 4096, 8x扩展)

# Model Architecture
vocab_size: 10
embed_dim: 1024
num_heads: 16
local_encoder_layers: 4
latent_encoder_layers: 20
latent_decoder_layers: 4
local_decoder_layers: 2
window_size: 16
dropout: 0.0
use_flash_attn: true
gradient_checkpointing: true

# === MergeDNA-Long新增 ===

# Entropy-Guided Merging
use_entropy_guided_merging: true
entropy_weight: 0.5
entropy_model_hidden_dim: 128
entropy_model_kernel_size: 9

# Learned Compression Schedule
use_learned_compression: true
r_min_per_window: 1
r_max_per_window: 8

# Hybrid Latent Encoder
latent_encoder_type: "hybrid"
ssm_type: "gated_deltanet"
attention_layer_indices: [5, 11, 17]   # 3层attention @ layers 6/12/18

# 更激进的压缩（32K→8K latent tokens，4x压缩）
compression_target: 0.25
compression_variance: 0.05

# Pre-training Objectives (同MergeDNA)
lambda_latent: 0.25
use_mtr: true
use_latent_mtr: true
use_amtm: true
amtm_masking_strategy: "adaptive"

# Training
max_steps: 100000
batch_size: 4                     # 长序列减小batch
gradient_accumulation: 8          # 有效batch = 32
learning_rate: 1.0e-4
weight_decay: 0.01
warmup_steps: 2000
max_grad_norm: 1.0
use_amp: true

# System
num_workers: 8
log_interval: 100
save_interval: 5000
output_dir: "./outputs/pretrain_long"
```

### 4.2 本地调试配置 `configs/pretrain_long_local.yaml`

```yaml
# MergeDNA-Long 本地调试（消费级GPU）
data_path: "./data/pretrain/multi_species_genomes"
max_seq_length: 4096              # 先用4K验证正确性

vocab_size: 10
embed_dim: 256                    # 缩小模型
num_heads: 4
local_encoder_layers: 2
latent_encoder_layers: 6
latent_decoder_layers: 2
local_decoder_layers: 1
window_size: 16
dropout: 0.0
use_flash_attn: false             # 本地可能没有flash-attn
gradient_checkpointing: false

# MergeDNA-Long
use_entropy_guided_merging: true
entropy_weight: 0.5
entropy_model_hidden_dim: 64
entropy_model_kernel_size: 5
use_learned_compression: true
r_min_per_window: 1
r_max_per_window: 8
latent_encoder_type: "hybrid"
ssm_type: "gated_deltanet"
attention_layer_indices: [2, 4]

compression_target: 0.3
compression_variance: 0.05
lambda_latent: 0.25
use_mtr: true
use_latent_mtr: true
use_amtm: true
amtm_masking_strategy: "adaptive"

max_steps: 500
batch_size: 2
gradient_accumulation: 1
learning_rate: 1.0e-4
weight_decay: 0.01
warmup_steps: 50
max_grad_norm: 1.0
use_amp: false

num_workers: 2
log_interval: 10
save_interval: 100
output_dir: "./outputs/pretrain_long_local"
max_samples: 1000
```

### 4.3 微调配置 `configs/finetune_long.yaml`

```yaml
# MergeDNA-Long Fine-tuning (A800)
pretrain_ckpt: "./outputs/pretrain_long/checkpoint-100000.pt"

# 模型参数必须与预训练一致
vocab_size: 10
embed_dim: 1024
num_heads: 16
local_encoder_layers: 4
latent_encoder_layers: 20
latent_decoder_layers: 4
local_decoder_layers: 2
window_size: 16
use_flash_attn: true
gradient_checkpointing: true

# MergeDNA-Long
use_entropy_guided_merging: true
entropy_weight: 0.5
entropy_model_hidden_dim: 128
entropy_model_kernel_size: 9
use_learned_compression: true
latent_encoder_type: "hybrid"
ssm_type: "gated_deltanet"
attention_layer_indices: [5, 11, 17]
compression_target: 0.25

# Fine-tuning
task_type: "sequence_classification"
max_seq_length: 32768
num_epochs: 10
batch_size: 16
learning_rate: 5.0e-5
weight_decay: 0.01
warmup_ratio: 0.1
max_grad_norm: 1.0
use_amp: true
use_lora: true
lora_rank: 8
lora_alpha: 16

# Data paths
genomic_benchmark_data_dir: "./data/genomic_benchmark_sequences"
nt_benchmark_data_dir: "./data/nt_benchmark"
gue_data_dir: "./data/gue_benchmark/GUE"
lrb_data_dir: "./data/lrb"
external_lrb_ref_dir: "./data/external_refs/genomics-long-range-benchmark"

# LRB specific
lrb_eqtl_sequence_length: 20000    # 原始MergeDNA通过滑窗处理，我们可以直接输入
lrb_bulk_rna_sequence_length: 40000
embedding_window_length: 32768      # 扩大窗口，减少滑窗次数
embedding_window_stride: 32768

output_dir: "./outputs/finetune_long"
```

---

## 五、实现步骤与运行命令

### Phase 0：环境准备与正确性验证

```bash
# 0.1 安装依赖
pip install flash-linear-attention

# 0.2 下载论文（手动或用脚本）
mkdir -p papers/plan_a
# 手动下载上述论文PDF到 papers/plan_a/

# 0.3 下载LRB数据
git clone https://github.com/Trop-LongRange/genomics-long-range-benchmark \
    ./data/external_refs/genomics-long-range-benchmark
mkdir -p ./data/lrb/raw
# 下载hg38.fa和hg19.fa（见上方命令）

# 0.4 验证现有代码能跑通（baseline）
python train.py --config configs/pretrain_local.yaml --mode pretrain --max_steps 50
```

### Phase 1：实现核心模块（第1-2周）

```bash
# 1.1 实现 entropy_model.py
# 1.2 修改 token_merging.py → EntropyGuidedTokenMerging
# 1.3 实现 hybrid_layers.py → GatedDeltaNetBlock
# 1.4 修改 latent_encoder.py → HybridLatentEncoder
# 1.5 修改 local_encoder.py → LearnedCompressionSchedule
# 1.6 修改 mergedna.py → MergeDNALongConfig + 新的forward逻辑

# 验证：本地smoke test
python train.py --config configs/pretrain_long_local.yaml --mode pretrain --max_steps 50
```

### Phase 2：效率验证（第2-3周）

```bash
# 2.1 效率对比脚本
python scripts/benchmark_efficiency.py \
    --configs configs/pretrain_a800.yaml configs/pretrain_long.yaml \
    --seq_lengths 4096 8192 16384 32768 65536 \
    --metrics flops memory throughput

# 预期结果：
#   seq=4K:  MergeDNA-Long ≈ MergeDNA（overhead from entropy model ~5%）
#   seq=32K: MergeDNA-Long 可运行，MergeDNA OOM
#   seq=32K: MergeDNA-Long memory ~40-55GB on A800 80GB
```

### Phase 3：预训练（第3-5周）

```bash
# 3.1 单GPU预训练（A800 80GB）
python train.py --config configs/pretrain_long.yaml --mode pretrain

# 3.2 多GPU预训练（如有8xA800）
torchrun --nproc_per_node=8 train.py \
    --config configs/pretrain_long.yaml --mode pretrain \
    --batch_size 4 --gradient_accumulation 1

# 预计时间：100K步 × ~8s/step ≈ 9天（单GPU）/ ~1.2天（8GPU）
```

### Phase 4：标准基准评估（第5-6周）

```bash
# 4.1 Genomic Benchmark (8任务)
python train.py --config configs/finetune_long.yaml \
    --mode finetune_all_gb --skip_existing

# 4.2 NT Benchmark (18任务)
python train.py --config configs/finetune_long.yaml \
    --mode finetune_all_nt --skip_existing

# 4.3 GUE Benchmark (24任务)
python train.py --config configs/finetune_long.yaml \
    --mode finetune_all_gue --skip_existing
```

### Phase 5：长程基准评估（第6-7周）

```bash
# 5.1 LRB Causal eQTL (20Kbp序列)
python -c "
from mergedna.experiments.lrb import run_lrb_eqtl
import yaml
config = yaml.safe_load(open('configs/finetune_long.yaml'))
run_lrb_eqtl(config, './outputs/lrb_long/eqtl')
"

# 5.2 LRB Bulk RNA Expression (40Kbp序列)
python -c "
from mergedna.experiments.lrb import run_lrb_bulk_rna
import yaml
config = yaml.safe_load(open('configs/finetune_long.yaml'))
run_lrb_bulk_rna(config, './outputs/lrb_long/bulk_rna')
"
```

### Phase 6：消融实验与分析（第7-8周）

```bash
# 6.1 消融：逐个关闭改进
# a) 关闭entropy-guided merging
python train.py --config configs/pretrain_long.yaml --mode pretrain \
    --max_steps 20000 --output_dir ./outputs/ablation/no_entropy
# 修改config: use_entropy_guided_merging: false

# b) 关闭hybrid encoder（用原始全注意力）
# 修改config: latent_encoder_type: "transformer"

# c) 关闭learned compression schedule
# 修改config: use_learned_compression: false

# d) 不同SSM类型
# 修改config: ssm_type: "mamba2"

# e) 不同attention层数/位置
# 修改config: attention_layer_indices: [9, 19]  (2层attention)
#             attention_layer_indices: [3, 7, 11, 15, 19]  (5层attention)

# 6.2 Tokenization可视化分析
python scripts/analyze_tokenization.py \
    --checkpoint ./outputs/pretrain_long/checkpoint-100000.pt \
    --sequences promoter enhancer splice_site intergenic \
    --output ./outputs/analysis/tokenization_viz.pdf
```

---

## 六、评估指标

### 6.1 标准基准（与MergeDNA论文对比）

| 基准 | 指标 | MergeDNA论文值 | MergeDNA-Long目标 |
|------|------|--------------|------------------|
| GB (8任务) | Top-1 Accuracy | 90.87% | ≥90.5%（不退化） |
| NT (18任务) | Average MCC/F1 | 78.39 | ≥78.0（不退化） |
| GUE (24任务) | Average MCC/F1 | 77.11% | ≥77.0（不退化） |

### 6.2 长程基准（核心创新验证）

| 基准 | 指标 | MergeDNA论文值 | 最佳Baseline | MergeDNA-Long目标 |
|------|------|--------------|-------------|------------------|
| LRB eQTL | AUROC | 0.75 | Evo2-7B: 0.74 | **≥0.78** |
| LRB Bulk RNA | R² | 0.62 | HybriDNA-7B: 0.60 | **≥0.65** |

### 6.3 效率指标

| 指标 | MergeDNA (4K) | MergeDNA-Long (32K) | 目标 |
|------|--------------|---------------------|------|
| 训练FLOPs/step | 1.0x | <2.0x（理想<1.5x） | 线性扩展 |
| 推理Memory | ~8GB | <60GB (A800) | 可运行 |
| 推理Throughput | 1.0x | >0.3x（8x长度） | 接近线性 |

### 6.4 消融指标

每个消融实验在GB 8任务上报告平均Accuracy：
- Full MergeDNA-Long
- - Entropy-Guided Merging
- - Hybrid Latent Encoder（用回全注意力）
- - Learned Compression Schedule
- - 所有改进（即原始MergeDNA）

---

## 七、预期结果与故事

### 论文核心贡献（按重要性排序）

1. **MergeDNA-Long架构**：首次将Token Merging与Hybrid SSM-Attention结合，在380M参数下实现32K-100K上下文的DNA建模
2. **Entropy-Guided Token Merging**：信息论驱动的合并策略，让tokenizer自适应分配分辨率
3. **在LRB长程基准上的SOTA**：380M参数超越7B的Evo2/HybriDNA（因为动态压缩的高效性）
4. **系统的效率分析**：展示Token Merging + SSM的scaling优势

### 预期的Figure/Table

1. **Figure 1**：MergeDNA-Long架构图（扩展原始MergeDNA，标注三个改进）
2. **Figure 2**：不同序列长度下的Memory/FLOPs对比（MergeDNA vs MergeDNA-Long vs Full-Attention vs SSM-only）
3. **Figure 3**：不同基因组区域的entropy-guided tokenization可视化（类似MergeDNA论文Fig.3但更informative）
4. **Figure 4**：学习到的逐层压缩率分布
5. **Table 1**：GB/NT/GUE标准基准对比（保持不退化）
6. **Table 2**：LRB长程基准对比（core result）
7. **Table 3**：消融实验
8. **Table 4**：效率对比（FLOPs, Memory, Throughput vs 序列长度）

### Rebuttal预案

| 可能的reviewer质疑 | 回应策略 |
|-------------------|---------|
| "与HybriDNA的区别是什么？" | HybriDNA是decoder-only AR模型，无token merging；我们是encoder-based autoencoder + 可学习动态tokenization |
| "entropy model增加了多少开销？" | ~1M参数（<0.3%），推理增加<5%延迟 |
| "为什么不直接用NTv3的U-Net压缩？" | NTv3用固定卷积下采样，我们用可学习的content-aware merging，消融实验证明优势 |
| "标准基准没有提升？" | 标准基准序列短(70bp-4Kbp)，长上下文改进在短序列上不应有显著提升，关键是不退化 |

---

## 八、时间规划

| 周 | 任务 | 产出 |
|----|------|------|
| 1 | 实现entropy_model.py + EntropyGuidedTokenMerging | 单元测试通过 |
| 2 | 实现hybrid_layers.py + HybridLatentEncoder + LearnedCompressionSchedule | 本地smoke test通过 |
| 3 | 效率benchmarking + 配置调优 | 效率对比Figure |
| 4-5 | 预训练（100K步） | 预训练checkpoint |
| 5-6 | GB/NT/GUE微调评估 | 标准基准Table |
| 6-7 | LRB长程评估 + SpliceAI/DMS评估 | 长程基准Table |
| 7-8 | 消融实验 + 可视化分析 | 消融Table + Figure |
| 8-9 | 论文撰写 | 初稿 |

**总计：~9周**（如有8xA800可缩短到~6周）
