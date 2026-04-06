# MergeDNA 项目总结

## 一、论文概述

**论文标题**: MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging

**核心问题**: 基因组DNA序列建模面临两大未解难题：
1. **信息密度不均匀** — 编码区(CDS)信息密集，而非编码区(nCDS)大量重复/低信息量，固定粒度的分词方案无法适应这种差异。
2. **缺乏明确的词边界** — 与自然语言不同，DNA没有天然的"单词"分界，有意义的单位（如密码子3bp、转录因子结合位点6-10bp）随上下文变化。

**核心思路**: 借鉴 Vision Transformer 中的 Token Merging (ToMe) 技术，构建一个层次化自编码器架构，将**可学习的动态分词器**与**全局上下文Transformer**联合训练。模型能根据输入序列的上下文自动决定哪些碱基该合并、合并成多长的token。

## 二、模型架构

MergeDNA 采用对称的自编码器（Autoencoder）架构，包含四个核心组件：

```
输入 X (N个碱基)
    │
    ▼
┌─────────────────┐
│  Local Encoder   │  ← 可学习分词器 (4层 LocalToMeAttention)
│  局部窗口注意力   │     每层: 局部注意力 + Token Merging
│  + Token合并     │     N → L tokens (约N/2)
└────────┬────────┘
         │ Z_L [B, L, D], Source矩阵 S [B, N, L]
         ▼
┌─────────────────┐
│  Latent Encoder  │  ← 主干网络 (20层 Full-Attention Transformer)
│  全局注意力       │     捕获长程依赖
└────────┬────────┘
         │ Z'_L [B, L, D]
         ▼
┌─────────────────┐
│  Latent Decoder  │  ← 轻量解码器 (4层 Transformer)
│  全局注意力       │     映射回token空间
└────────┬────────┘
         │ Z_hat_L [B, L, D]
         ▼
┌─────────────────┐
│  Local Decoder   │  ← 反分词器 (2层局部窗口注意力)
│  Token Unmerge   │     用Source矩阵恢复原始长度
│  + 局部注意力     │     L → N, 输出重建序列 X_hat
└─────────────────┘
```

### 2.1 Local Encoder — 可学习分词器 (`local_encoder.py`)

- 逐碱基嵌入后，堆叠4层 `LocalToMeAttentionLayer`
- 每层内先做**局部窗口自注意力**（窗口大小 w=16），再做**可微分Token合并**
- Token合并机制：在每个窗口内，用轻量分组嵌入(DTEM)计算相邻token对的相似度，选top-r对进行**软合并**（加权平均），同时更新Source矩阵 S 追踪合并历史
- 经过4层后，序列从 N 压缩到 L（约 N/2），Source矩阵 S ∈ {0,1}^{N×L} 记录每个原始位置属于哪个合并后的token

### 2.2 Latent Encoder — 全局上下文建模 (`latent_encoder.py`)

- 20层标准Transformer (LLaMA风格: RMSNorm + RoPE + SwiGLU + Flash Attention)
- 对L个合并后的token做全局注意力，捕获长程依赖
- 预训练时额外支持 `forward_with_selection`：使用全局ToMe选出K个最重要的token（用于计算Latent MTR loss）

### 2.3 Latent Decoder (`latent_encoder.py`)

- 4层Transformer，与Latent Encoder对称但更轻量
- 将上下文增强后的表示 Z'_L 映射回token空间

### 2.4 Local Decoder — 反分词器 (`local_decoder.py`)

- **Token Unmerge**: 利用Source矩阵 S 将L个token恢复为N个碱基位置的表示（矩阵乘法 Z_N = S @ Z_hat_L）
- 2层局部窗口注意力精炼局部细节
- 输出头预测每个位置的碱基（vocab_size=10的分类）

### 2.5 Token Merging 机制 (`token_merging.py`)

两种Token Merging:

| 类型 | 使用位置 | 作用 |
|------|---------|------|
| `LocalWindowTokenMerging` | Local Encoder | 在局部窗口内合并相似的相邻token，实现动态分词 |
| `GlobalTokenMerging` | Latent Encoder (预训练) | 全局选取K个最重要token，用于自适应掩码策略 |

核心算法：二部图软匹配 (Bipartite Soft Matching)
1. 将token分为偶数位(set A)和奇数位(set B)
2. 计算A-B间的余弦相似度
3. 选最相似的r对进行合并：A中的"被合并者"的表示加权平均到B中的"保留者"上
4. 更新Source矩阵追踪合并关系

## 三、预训练目标

三个损失函数联合训练（需要三次前向传播）：

### 3.1 L_MTR — 合并Token重建 (Merged Token Reconstruction)
- 完整自编码器前向：输入 → Local Encoder → Latent Encoder → Latent Decoder → Local Decoder → 重建
- 交叉熵损失：重建序列 X_hat 与原始输入 X 的逐碱基交叉熵

### 3.2 Latent L_MTR — 潜在空间重建 (λ=0.25)
- **冻结Local Encoder**，只训练Latent Encoder/Decoder
- Latent Encoder通过全局ToMe选K个token → Latent Decoder → Unmerge → Local Decoder → 重建
- 迫使模型学会仅从K个token压缩重建整个序列

### 3.3 L_AMTM — 自适应掩码Token建模 (Adaptive Masked Token Modeling)
- 利用全局ToMe的合并结果S'构建自适应掩码：**小group的token（信息量高）被掩码的概率高，大group的token（重复/低信息）被掩码的概率低**
- 对掩码后的输入走完整自编码器，只在被掩码位置计算损失
- 引导模型关注功能相关的信息密集区域

**总损失**: L_total = L_MTR + 0.25 × L_latent_MTR + L_AMTM

## 四、项目代码结构

```
MergeDNA/
├── mergedna/                          # 核心Python包
│   ├── model/
│   │   ├── mergedna.py                # 主模型 (MergeDNA, 分类/token分类包装)
│   │   ├── local_encoder.py           # Local Encoder (可学习分词器)
│   │   ├── latent_encoder.py          # Latent Encoder + Latent Decoder
│   │   ├── local_decoder.py           # Local Decoder + Token Unmerge
│   │   ├── token_merging.py           # LocalWindow/Global Token Merging
│   │   └── transformer.py            # Transformer基础组件 (RMSNorm, RoPE, SwiGLU, MHA)
│   ├── data/
│   │   ├── dataset.py                 # 数据集 (Multi-Species, Genomic/NT/GUE Benchmark)
│   │   ├── tokenizer.py              # 字符级DNA Tokenizer (A=5,T=6,C=7,G=8,N=9)
│   │   └── collator.py               # 数据Collator (预训练/微调)
│   ├── training/
│   │   ├── pretrain.py                # 预训练Runner (AdamW, cosine schedule, DDP)
│   │   ├── finetune.py               # 微调Runner (LoRA, 分类/token分类)
│   │   └── losses.py                 # 预训练损失 (L_MTR + Latent L_MTR + L_AMTM)
│   └── utils/
│       └── utils.py
├── train.py                           # 训练入口 (预训练/微调/全benchmark评测)
├── evaluate.py                        # 评估脚本
├── configs/                           # YAML配置文件
│   ├── pretrain_a800.yaml             # A800全量预训练 (380M, 100K steps)
│   ├── pretrain_local.yaml            # 本地小规模预训练
│   ├── pretrain_quicktest.yaml        # 快速测试
│   ├── finetune_a800.yaml             # A800微调配置
│   ├── finetune_genomic_benchmark.yaml
│   ├── finetune_nt_benchmark.yaml
│   └── finetune_local.yaml
├── scripts/
│   ├── core_idea_walkthrough.py       # 教学脚本：逐步演示模型核心流程
│   ├── download_data.py               # 数据下载
│   ├── run_a800.sh / pretrain.sh      # 训练启动脚本
│   └── finetune_*.sh                  # 微调启动脚本
└── data/
    └── gue_benchmark/GUE/             # GUE基准数据 (24个子任务)
```

## 五、论文设定 vs 代码实现对照

| 论文参数 | 论文值 | 代码对应 | 代码值 |
|---------|--------|---------|--------|
| 总参数量 | 380M | `MergeDNAConfig` 默认值组合 | ~380M |
| 嵌入维度 D | 1024 | `embed_dim` | 1024 |
| 注意力头 | 16 | `num_heads` | 16 |
| 局部窗口 w | 16 | `window_size` | 16 |
| Local Encoder层数 | 4 | `local_encoder_layers` | 4 |
| Latent Encoder层数 | 20 | `latent_encoder_layers` | 20 |
| Latent Decoder层数 | 4 | `latent_decoder_layers` | 4 |
| Local Decoder层数 | 2 | `local_decoder_layers` | 2 |
| 最大序列长度 | 4096 | `max_seq_length` | 4096 |
| 压缩目标 L/N | ~0.5 | `compression_target` | 0.5 |
| λ (latent loss权重) | 0.25 | `lambda_latent` | 0.25 |
| 预训练步数 | 100K | `max_steps` | 100000 |
| 学习率 | 1e-4 | `learning_rate` | 1e-4 |
| 优化器 | AdamW | `torch.optim.AdamW` | AdamW(β1=0.9, β2=0.95) |
| 预训练数据 | Multi-Species Genomes | `MultiSpeciesGenomeDataset` | 同 |
| 微调方式 | LoRA | `apply_lora(rank=8, alpha=16)` | rank=8, alpha=16 |

## 六、支持的Benchmark

| Benchmark | 任务数 | 任务类型 | 评估指标 |
|-----------|-------|---------|---------|
| Genomic Benchmark | 8 | 增强子/物种分类/调控元件 | Top-1 Accuracy |
| NT Benchmark | 18 | 组蛋白标记/增强子/启动子/剪接位点 | MCC / F1 |
| GUE Benchmark | 24+ | 表观/转录因子/启动子/病毒/剪接位点 | MCC / F1 |

## 七、运行方式

```bash
# 预训练 (单GPU A800)
python train.py --config configs/pretrain_a800.yaml --mode pretrain

# 微调单任务
python train.py --config configs/finetune_a800.yaml --mode finetune --task_name H3

# 微调全部Genomic Benchmark (8个任务)
python train.py --config configs/finetune_a800.yaml --mode finetune_all_gb

# 微调全部NT Benchmark (18个任务)
python train.py --config configs/finetune_a800.yaml --mode finetune_all_nt

# 微调全部GUE Benchmark (24+个任务)
python train.py --config configs/finetune_a800.yaml --mode finetune_all_gue

# 评估
python evaluate.py --config configs/finetune_a800.yaml \
    --checkpoint ./outputs/finetune/best_model.pt \
    --benchmark genomic_benchmark

# 核心流程演示
python scripts/core_idea_walkthrough.py
```

## 八、关键技术亮点

1. **端到端动态分词**: Local Encoder通过可微分Token Merging自动学习DNA的分词边界，不依赖预定义的k-mer或BPE。在不同基因组上下文（剪接位点、启动子、增强子）产生不同长度的token分布。

2. **层次化压缩**: Local Encoder (局部注意力, O(N·w)) → Latent Encoder (全局注意力, O(L²)) 的两阶段设计，在保持全局建模能力的同时控制计算复杂度。

3. **信息感知的预训练**: AMTM损失让模型自适应地关注信息密集区域（被小group合并的token），而非像标准MLM那样均匀随机掩码，避免在重复区域浪费学习容量。

4. **Source矩阵追踪**: 通过维护 Source 矩阵 S，实现了Token合并与反合并的精确对应，使得解码器能准确恢复原始碱基分辨率。

