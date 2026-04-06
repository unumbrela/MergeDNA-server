# MergeDNA 项目概览

> 本项目是对论文 **"MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging"** (Li et al., AAAI 2026) 的复现实现，结合了多个参考仓库（`reference_repos/`）的代码设计。

---

## 1. 论文核心思想

### 1.1 解决的问题

DNA序列建模面临两大未解决的挑战：
1. **信息密度分布不均**：基因组中编码区(CDS, ~2%)信息密集，非编码区(nCDS)多为重复/低信息内容。固定token化（如k-mer、BPE）无法自适应调节粒度。
2. **缺乏天然词边界**：DNA不像自然语言有空格/标点，有意义的"单词"可能是3碱基的密码子、6-10碱基的转录因子结合位点，或更长的序列。

 创新点来源与思路链

  核心灵感：从CV领域的 Token Merging (ToMe, Bolya et al. ICLR 2023) + NLP领域的 Byte Latent Transformer (BLT)
  的entropy-based patching，迁移到基因组学。

  为什么能work：DNA有一个独特性质——信息密度极度不均匀（coding region ~2% vs.
  大量重复/非编码区），但又没有自然的"词边界"。固定tokenization（BPE、k-mer）对此无能为力。MergeDNA的关键洞察：

  1. 让模型自己学"词边界"：Local Encoder通过可微分的token merging，在local
    window内根据相似性合并相邻base，信息密集区保留细粒度，重复区aggressive合并
    2.
    联合优化tokenizer和encoder：不像BPE/VQDNA先训tokenizer再训模型，而是end-to-end，tokenization直接受预训练目标的梯度更新
  3. 三个互补的预训练目标：MTR训全pipeline，Latent MTR迫使latent encoder从更少token恢复信息（类似information
    bottleneck），AMTM让模型聚焦高信息位置（merge后的singleton/小group）

### 1.2 核心方案：可学习的动态Token化 + 上下文感知预训练

MergeDNA将Token Merging（ToMe）技术从Vision Transformer迁移到DNA基因组建模，构建了一个**层次化自编码器**架构：

```
Input X (N bases)
    |
    v
[Local Encoder] -- 可学习tokenizer，通过堆叠的local-window attention + token merging
    |                逐层合并相邻碱基为"词"，输出 Z_L (L tokens) + Source Matrix S
    v
[Latent Encoder] -- 全注意力Transformer，捕获全局长程依赖
    |                输出 Z'_L (L tokens, 上下文增强)
    v
[Latent Decoder] -- 轻量Transformer，将潜在表示映射回token空间
    |                输出 Z_hat_L
    v
[Local Decoder]  -- Token Unmerging (利用Source Matrix S恢复长度N)
    |                + local-window attention精炼局部细节
    v
Output X_hat (N bases) -- 重建序列
```

### 1.3 关键技术细节

#### Token Merging机制（论文 Section 3.3）
- **Local-window Token Merging**：在Local Encoder中，每层在局部窗口(w=16)内：
  1. 通过DTEM（轻量投影层，D -> D/4）计算token间相似度
  2. 二部图软匹配(bipartite soft matching)：将token分为奇/偶两组，找最相似配对
  3. 软合并：keeper吸收merged token（加权平均），更新Source Matrix
  4. 每层减少r个token/窗口，4层后从N压缩到约L≈N/2

- **Global Token Merging**：在Latent Encoder预训练阶段，进一步从L个token中选出K个关键token（K≈L/2），用于latent MTR loss和自适应掩码策略。

#### Source Matrix S
- `S ∈ {0,1}^{N×L}`：记录每个原始碱基位置属于哪个合并后的token
- 初始为单位矩阵 I_N，每层合并后更新
- **用途**：Token Unmerging时通过 `Z_N = S @ Z_hat_L` 恢复原始长度；AMTM中将token空间掩码映射回碱基空间

#### 三个预训练目标（论文 Eq. 8）

```
L_total = L_MTR + λ * L_latent_MTR + L_AMTM    (λ=0.25)
```

1. **L_MTR（Merged Token Reconstruction）**：标准自编码器重建loss，从压缩token重建原始序列，训练整个pipeline包括Local Encoder
2. **Latent L_MTR（Local Encoder冻结）**：冻结Local Encoder，Latent Encoder通过Global ToMe选K个token后重建。迫使latent model学会从更少的token中恢复信息
3. **L_AMTM（Adaptive Masked Token Modeling）**：基于合并结果S'的自适应掩码策略——大合并组（低信息）的token掩码概率低，小组/单独token（高信息）掩码概率高。聚焦于预测信息丰富的token

#### 压缩率随机采样
训练时L不固定，从高斯分布采样 `L ~ N(N/2, σ)`，范围限制在[0.4N, 0.6N]，防止过拟合到特定压缩率。

---

## 2. 项目代码结构

```
MergeDNA/
├── train.py                    # 统一入口：pretrain / finetune / finetune_all_*
├── evaluate.py                 # 评估脚本
├── mergedna/
│   ├── model/
│   │   ├── transformer.py      # LLaMA风格基础组件：RMSNorm, RoPE, SwiGLU, MHA, TransformerBlock, LocalWindowAttention
│   │   ├── token_merging.py    # LocalWindowTokenMerging + GlobalTokenMerging（核心创新）
│   │   ├── local_encoder.py    # Local Encoder = stack of (LocalWindowAttn + TokenMerge)
│   │   ├── latent_encoder.py   # Latent Encoder (20层全注意力) + Latent Decoder (4层全注意力)
│   │   ├── local_decoder.py    # Local Decoder = TokenUnmerge + LocalWindowAttn + OutputHead
│   │   └── mergedna.py         # MergeDNA主模型 + 分类/token分类封装
│   ├── data/
│   │   ├── tokenizer.py        # DNACharTokenizer：字符级{A,T,C,G,N} -> token ID
│   │   ├── collator.py         # PretrainCollator / FineTuneCollator
│   │   └── dataset.py          # MultiSpeciesGenomeDataset, GenomicBenchmarkDataset, NTBenchmarkDataset, GUEBenchmarkDataset
│   ├── training/
│   │   ├── losses.py           # MergeDNAPretrainLoss：L_MTR + λ*L_latent_MTR + L_AMTM
│   │   ├── pretrain.py         # PretrainRunner：DDP/AMP/梯度累积/cosine schedule
│   │   └── finetune.py         # FineTuneRunner：LoRA微调 + 评估
│   └── experiments/            # 扩展实验：SpliceAI, LRB, Protein Fitness, Ablation
│       ├── spliceai.py
│       ├── lrb.py
│       ├── protein_fitness.py
│       ├── ablation.py
│       └── common.py
├── configs/
│   ├── pretrain_a800.yaml      # A800 80GB 全规模预训练配置（论文设置）
│   ├── pretrain_local.yaml     # 本地小规模调试配置
│   ├── finetune_a800.yaml      # A800 微调配置
│   ├── finetune_local.yaml     # 本地微调调试
│   ├── finetune_genomic_benchmark.yaml
│   └── finetune_nt_benchmark.yaml
├── scripts/
│   ├── download_data.py        # 数据下载（Multi-Species Genomes, GUE, NT, GB等）
│   ├── run_all_a800.sh         # A800全流程脚本
│   ├── run_all_local.sh        # 本地全流程脚本
│   ├── run_all_experiments.py  # 全部实验自动化
│   └── core_idea_walkthrough.py  # 核心思想演示脚本
├── reference_repos/            # 参考实现
│   ├── nucleotide-transformer/ # NT基准参考
│   ├── DNABERT_2/              # DNABERT-2预训练数据格式参考
│   ├── VQDNA/                  # 动态tokenizer参考
│   ├── MxDNA/                  # 动态tokenizer参考
│   ├── ToMe/                   # Token Merging原始实现（ViT）
│   ├── blt/                    # Byte Latent Transformer参考
│   ├── hnet/                   # HNet可微分分割参考
│   ├── hyena-dna/              # 长序列DNA模型参考
│   ├── mamba/                  # SSM参考
│   └── ...
└── docs/                       # 文档
```

---

## 3. 模型架构详细参数（论文 Table A1）

| 组件 | 层数 | Attention类型 | 参数量 |
|------|------|--------------|--------|
| Local Encoder | 4 | Local-window (w=16) | ~51M |
| Latent Encoder | 20 | Full Attention (Flash) | ~253M |
| Latent Decoder | 4 | Full Attention (Flash) | ~51M |
| Local Decoder | 2 | Local-window (w=16) | ~25M |
| **总计** | **30** | - | **~380M** |

- Embedding dim: 1024
- Attention heads: 16
- FFN: SwiGLU, hidden_dim ≈ 8/3 * 1024 ≈ 2730（对齐到256的倍数）
- 位置编码: RoPE
- 归一化: RMSNorm (Pre-norm)

---

## 4. 训练流程

### 4.1 预训练

- **数据集**：Multi-Species Genomes（多物种基因组参考序列RefSeq，来自NCBI）
- **优化器**：AdamW, β=(0.9, 0.95), lr=1e-4, weight_decay=1e-8(论文) / 0.01(实现)
- **调度**：Cosine annealing + 2000步warmup
- **迭代**：100K步
- **批大小**：8 per GPU × 4 gradient accumulation = 有效批大小32
- **序列长度**：4096
- **精度**：BFloat16 混合精度

**三次前向传播**（每步）：
1. 完整autoencoder前向 → L_MTR
2. 冻结Local Encoder + Global ToMe选K token → Latent L_MTR
3. 自适应掩码输入 → L_AMTM

### 4.2 微调

- **策略**：LoRA (rank=8, alpha=16) 应用于Latent Encoder的QKV/O投影
- **任务类型**：
  - 序列分类（编码器-only模式）：丢弃decoder，在Latent Encoder输出上接MLP分类头
  - Token分类（编码器-解码器模式）：保留decoder恢复序列分辨率，接token级分类头
- **优化器**：AdamW, lr∈{1e-5, 5e-5, 1e-4}, batch=32, 10 epochs

---

## 5. 评估基准

### 5.1 已实现的基准

| 基准 | 任务数 | 评估指标 | 代码位置 |
|------|--------|---------|---------|
| Genomic Benchmark (GB) | 8 | Top-1 Accuracy | `GenomicBenchmarkDataset` |
| Nucleotide Transformer (NT) | 18 | MCC / F1 | `NTBenchmarkDataset` |
| GUE (Genome Understanding) | 24 | MCC / F1 | `GUEBenchmarkDataset` |
| SpliceAI | 2 (donor/acceptor) | AUROC | `experiments/spliceai.py` |
| LRB (Long-Range Benchmark) | 2 (eQTL/Bulk RNA) | AUROC / R² | `experiments/lrb.py` |
| Protein Fitness (DMS) | 2 | SRCC | `experiments/protein_fitness.py` |

### 5.2 论文报告的核心结果

- **GB 8任务平均Top-1准确率**：90.87%（SOTA）
- **NT 18任务平均MCC/F1**：78.39（SOTA）
- **GUE 24任务平均**：77.11%（SOTA）
- **SpliceAI平均AUROC**：69.8
- **LRB eQTL AUROC**：0.75, Bulk RNA R²：0.62（均SOTA）

---

## 6. 使用方式

### 预训练
```bash
# 单GPU (A800 80GB)
python train.py --config configs/pretrain_a800.yaml --mode pretrain

# 多GPU (torchrun)
torchrun --nproc_per_node=8 train.py --config configs/pretrain_a800.yaml --mode pretrain
```

### 微调
```bash
# 单任务
python train.py --config configs/finetune_a800.yaml --mode finetune --task_name H3

# 全部Genomic Benchmark (8任务)
python train.py --config configs/finetune_a800.yaml --mode finetune_all_gb

# 全部NT Benchmark (18任务)
python train.py --config configs/finetune_a800.yaml --mode finetune_all_nt

# 全部GUE Benchmark (24+任务)
python train.py --config configs/finetune_a800.yaml --mode finetune_all_gue --gue_data_dir ./data/gue_benchmark/GUE
```

### 评估
```bash
python evaluate.py --config configs/finetune_a800.yaml \
    --checkpoint ./outputs/finetune/best_model.pt \
    --benchmark genomic_benchmark
```

---

## 7. 参考仓库与复现关系

| 参考仓库 | 在本项目中的作用 |
|---------|----------------|
| `ToMe` | Token Merging核心算法（bipartite soft matching）的原始实现来源 |
| `blt` (Byte Latent Transformer) | 层次化byte-level架构设计参考 |
| `hnet` | 可微分分割/chunking的设计参考 |
| `DNABERT_2` | Multi-Species Genomes预训练数据格式、GUE基准数据 |
| `nucleotide-transformer` | NT基准任务定义、评估协议 |
| `VQDNA` / `MxDNA` | 动态DNA tokenizer的对比方法参考 |
| `hyena-dna` / `mamba` | 长序列DNA建模的SSM方法参考 |
| `Mixture-of-depths` | token选择/路由策略参考 |

---

## 8. 关键设计决策与实现注意事项

1. **字符级输入**：使用byte-level tokenizer（A/T/C/G/N → token ID），vocab_size=10（含PAD/CLS/SEP/MASK/UNK特殊token）。与BPE/k-mer不同，tokenization完全由模型学习。

2. **Source Matrix内存**：S ∈ R^{B×N×L}，当N=4096时占用大量内存。实现中source作为float类型传播，用于反向传播梯度。

3. **固定记录数据集**：预训练数据（DNABERT-2格式，每行1000bp）支持O(1)随机访问，避免将大文件加载到内存。

4. **梯度检查点**：对Latent Encoder的20层全注意力块启用gradient checkpointing，节省约60%激活内存。

5. **Flash Attention**：Latent Encoder/Decoder使用Flash Attention加速全注意力计算；Local Encoder/Decoder的窗口注意力使用标准实现（窗口大小仅16）。

6. **预训练三次前向传播**：每步需要3次前向传播（MTR, latent MTR, AMTM），训练成本约为普通MLM的3倍。

7. **Skip existing**：finetune_all_*模式支持`--skip_existing`参数，跳过已完成的任务（检查results.json）。
