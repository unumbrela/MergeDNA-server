# MergeDNA 创新方向调研报告

> 目标：基于MergeDNA项目，寻找可向NeurIPS投稿的创新/改进方向。
> 调研时间：2026-04-06
> 调研范围：DNA基础模型、动态tokenization、高效架构、多组学模型、NeurIPS 2024-2025趋势

---

## 调研总结：当前领域格局

MergeDNA (AAAI 2026) 的核心贡献是将Token Merging引入DNA建模，实现了可学习的动态tokenizer。但以下**关键gap**仍然存在：

1. **上下文长度受限**：MergeDNA仅支持4096bp，而Evo2(1M)、Genos(1M)、NTv3(1Mb)已推进到百万碱基级别
2. **Latent Encoder效率瓶颈**：20层全注意力Transformer是计算瓶颈，限制了序列扩展
3. **合并决策与表示耦合**：当前用同一embedding做attention和merge决策，存在目标冲突
4. **缺乏生物学可解释性**：未验证学到的token边界是否对应真实的生物学单元
5. **单模态限制**：仅处理DNA序列，未连接RNA/蛋白质等下游模态

---

## 方向一：Entropy-Guided Adaptive Token Merging（熵引导的自适应合并）

### 故事线
MergeDNA的Token Merging基于学习的相似度进行合并，但缺乏对"信息量"的显式建模。BLT（Byte Latent Transformer, Meta 2024）提出entropy-based patching——在信息熵高的位置设置分段边界，高熵区用细粒度、低熵区用粗粒度。DNA序列天然适合这一思路：保守区域（低熵）可大幅合并，高变异区域（高熵，如调控元件、splice sites）需要保留细粒度。

### 核心Idea
**将局部熵估计作为Token Merging的先验信号**：训练一个轻量的熵预测模块（小的causal LM或简单的n-gram统计），在合并前评估每个位置的信息量，然后将熵分数注入merge similarity计算中，使高熵区域的token"抵抗"合并。

```
改进后的merge score = similarity_score - α * entropy_score
```

### 关键论文与资源
| 资源 | 用途 |
|------|------|
| **BLT** (Yu et al., Meta 2024, arXiv:2412.09871) | 熵引导分段的核心方法 |
| `reference_repos/blt/` | 已有BLT参考实现 |
| **HNet** (Hwang et al., 2025, arXiv:2507.07955) | 可微分边界预测的替代方案 |
| `reference_repos/hnet/` | 已有HNet参考实现 |

### 实现路径
1. 在`mergedna/model/token_merging.py`的`LocalWindowTokenMerging`中，增加entropy-aware scoring
2. 训练一个轻量的byte-level entropy model（2-3层小Transformer），在Local Encoder之前预测每个碱基的局部熵
3. 将entropy score融入bipartite matching的similarity计算
4. 可选：用HNet的可微分边界预测替代bipartite matching

### 验证方案
- **消融实验**：在GB/NT基准上对比原始ToMe merge vs entropy-guided merge
- **可解释性分析**：可视化不同基因组区域（promoter/enhancer/splice site/intergenic）的合并粒度分布，验证熵引导的合并是否更好地匹配生物学功能区域
- **压缩率-性能曲线**：在不同压缩率下，entropy-guided方法应在高压缩率时保持更好的性能

### 创新性评估
- **新颖度**：中高。BLT的entropy patching用于NLP，但从未应用于DNA建模，且与可微分token merging的结合是全新的
- **实现难度**：中等。主要修改token_merging.py和local_encoder.py
- **NeurIPS适合度**：高。方法论清晰，有明确的生物学动机

---

## 方向二：Hybrid SSM-Attention Latent Encoder（混合SSM-注意力架构）

### 故事线
MergeDNA的20层全注意力Latent Encoder是参数和计算的主要来源（253M/380M）。最新研究表明，**混合SSM-Attention架构**以6:1的比例（6层SSM + 1层Attention）在DNA建模上超越纯Transformer（HybriDNA, 2025）。Gated DeltaNet（ICLR 2025）和Mamba-3（ICLR 2026）提供了更强的线性复杂度替代方案。关键洞察：Token Merging + SSM是一个被忽视的组合——合并后的token序列天然更短更紧凑，非常适合SSM的O(N)处理。

### 核心Idea
**MergeDNA + Hybrid Latent Encoder = 长上下文DNA建模的高效方案**

将20层全注意力替换为：
- 17层 Gated DeltaNet（或Mamba-3）：O(N)复杂度处理大部分层
- 3层 Full Attention（置于层6/12/18）：保留全局recall能力
- 结合MergeDNA的~50%压缩率，100K输入 → ~50K latent tokens，主要以线性复杂度处理

### 关键论文与资源
| 资源 | 用途 |
|------|------|
| **Gated DeltaNet** (Yang et al., ICLR 2025, arXiv:2412.06464) | 最强线性注意力变体 |
| **Mamba-3** (ICLR 2026) | 最新SSM，复数值状态空间 |
| **HybriDNA** (Ma et al., 2025, arXiv:2502.10807) | 混合架构在DNA上的验证 |
| **NTv3** (InstaDeep, Dec 2025) | U-Net压缩+Transformer的成功先例 |
| `reference_repos/mamba/` | Mamba参考实现 |
| `flash-linear-attention` (GitHub) | GLA/DeltaNet高效CUDA核 |
| **MaMe** (arXiv:2508.13599) | SSM专用的token merging策略 |

### 实现路径
1. `pip install flash-linear-attention` 或 `pip install mamba-ssm`
2. 在`mergedna/model/latent_encoder.py`中新增`HybridLatentEncoder`，交替使用DeltaNet/Mamba层和Attention层
3. 修改`MergeDNAConfig`增加`latent_encoder_type: "hybrid"`, `ssm_type: "gated_deltanet"`, `attention_ratio: 0.15`
4. 扩展`max_seq_length`到32K-100K，验证长上下文能力

### 验证方案
- **效率对比**：相同参数量下，对比原始全注意力 vs 混合架构的训练速度/内存占用
- **长上下文基准**：在LRB（eQTL 20Kbp, Bulk RNA 40Kbp）上验证长程建模能力
- **标准基准保持**：确认GB/NT/GUE上性能不退化
- **Scaling曲线**：展示随序列长度增长，混合架构的效率优势

### 创新性评估
- **新颖度**：中高。"Token Merging + SSM"的组合未被探索过，且MaMe论文（2025.08）才刚出现
- **实现难度**：中等。需要集成外部SSM库
- **NeurIPS适合度**：高。效率+性能的story NeurIPS很看重

---

## 方向三：Decoupled Token Merge Embeddings + Learned Compression Schedule

### 故事线
DTEM（Lee & Hong, NeurIPS 2024）揭示了一个关键问题：在标准ToMe中，用于语义计算的embedding同时被用来计算merge相似度，产生**目标冲突**。此外，DiffRate（Chen et al., ICCV 2023）表明固定的逐层压缩率是次优的——应该学习每层的压缩率。在DNA场景中这一问题更严重：DNA的碱基embedding同时需要编码序列身份和生物功能，合并决策不应干扰表示学习。

### 核心Idea
**双轨Token Merging**：
1. 解耦的merge embedding：一个独立的投影头专门学习"哪些token应该合并"，与主表示分离
2. 学习的逐层压缩调度：通过Gumbel-Softmax学习每层应该合并多少token，而非预设固定的r_per_window
3. 辅助损失：merge embedding通过一个辅助重建损失训练，确保合并后的token忠实代表被丢弃的token

### 关键论文与资源
| 资源 | 用途 |
|------|------|
| **DTEM** (Lee & Hong, NeurIPS 2024) | 解耦merge embedding的核心方法 |
| **DiffRate** (Chen et al., ICCV 2023) | 可微分压缩率调度 |
| **PaCa** (Liang et al., 2023) | Clustering-based merge替代方案 |

### 实现路径
1. 在`token_merging.py`的`LocalWindowTokenMerging`中，将`group_proj`替换为独立的DTEM模块
2. 增加可学习的`r_schedule`参数（每层一个），通过Gumbel-Softmax在训练中自动学习
3. 增加辅助损失：`L_merge = ||merge_embedding(merged_token) - mean(merge_embedding(constituent_tokens))||`

### 验证方案
- **消融实验**：DTEM vs 原始shared embedding在GB/NT上的对比
- **压缩率可视化**：学习到的逐层压缩率是否有意义（如早期层少合并保留局部信息，后期层多合并提取全局模式）
- **不同基因组区域的adaptive behavior**：在promoter/enhancer/intergenic区域，DTEM是否学到了不同的合并策略

### 创新性评估
- **新颖度**：中。DTEM本身已有，但应用于DNA Token Merging + 可学习压缩调度的组合是新的
- **实现难度**：低-中。主要修改token_merging.py
- **NeurIPS适合度**：中。作为独立工作可能不够，适合与其他方向组合

---

## 方向四：Mixture-of-Depths for Genomic Tokens（基因组Token的深度混合）

### 故事线
基因组序列中，绝大多数碱基从局部上下文就可以预测（如保守基因内部），只有少数功能关键位置（SNP、调控元件边界、splice junction）需要深层推理。Mixture-of-Depths（MoD, Google DeepMind 2024）允许每层的router决定哪些token参与full computation，哪些直接skip。Mixture-of-Recursions（MoR）进一步允许token被同一层处理多次。

### 核心Idea
**在MergeDNA的Latent Encoder中引入MoD/MoR**：合并后的latent tokens不是每层都做full attention，而是通过learned router选择top-k token参与计算。"简单"的合并token（对应低信息区域）可以skip，"重要"的合并token（对应功能区域）获得完整的计算深度。

进一步结合token merging的信息：可以用merge group size（Source Matrix S中的group size）作为router的先验——大group的token（合并了很多碱基）更可能是低信息的，应该skip。

### 关键论文与资源
| 资源 | 用途 |
|------|------|
| **Mixture-of-Depths** (Raposo et al., Google DeepMind, 2024) | 核心方法 |
| `reference_repos/Mixture-of-depths/` | 已有MoD参考实现 |
| **Mixture-of-Recursions** | 自适应递归深度 |
| `reference_repos/mixture_of_recursions/` | 已有MoR参考实现 |

### 实现路径
1. 在`TransformerBlock`中增加MoD router（小Linear层输出每个token的routing score）
2. 每层只处理top-k%的token（如top-50%），其余通过residual connection直接传递
3. 可选：用merge group size作为router的额外输入特征
4. 训练时用auxiliary load-balancing loss确保router不退化

### 验证方案
- **效率**：相同准确率下FLOPs减少多少
- **Router分析**：可视化哪些token被skip、哪些被选中，是否与基因组功能区域对应
- **长序列scaling**：MoD在长序列（>4K）上的加速效果更显著

### 创新性评估
- **新颖度**：中高。MoD在NLP/Vision中已有，但与DNA Token Merging的结合（merge信息指导routing）是全新的
- **实现难度**：中等
- **NeurIPS适合度**：中高。"自适应计算分配"在基因组中的生物学动机很强

---

## 方向五：Cross-Modal Dynamic Tokenization（跨模态动态Token化）

### 故事线
当前生物基础模型正在向中心法则（DNA→RNA→蛋白质）统一方向发展（Life-Code, LucaOne, CDBridge, AIDO）。这些模型都使用固定tokenization，但DNA/RNA/蛋白质有不同的"天然词单元"——DNA中的密码子(3bp)、RNA的二级结构元素(可变)、蛋白质的domain(可变)。MergeDNA的可学习Token Merging天然适合作为**跨模态动态tokenizer**。

### 核心Idea
**统一动态Tokenizer for Central Dogma**：
1. 训练一个共享的Local Encoder，输入可以是DNA/RNA序列（字符级），通过Token Merging学习模态特异的分段策略
2. DNA模态：学习密码子边界（CDS区域合并为3-mer）、调控元件边界
3. RNA模态：学习茎环结构边界、剪接位点
4. 共享的Latent Encoder在统一的latent space中建模
5. 预训练目标扩展：MTR + 跨模态对齐损失（DNA编码→RNA编码的对应关系）

### 关键论文与资源
| 资源 | 用途 |
|------|------|
| **Life-Code** (Liu et al., 2025, arXiv:2502.07299) | 反转录统一表示 |
| **CDBridge** (OpenReview 2025) | 剪接感知Token Merging + 跨模态桥接 |
| **LucaOne** (Nature Machine Intelligence 2025) | 统一核酸/蛋白质模型 |
| **AIDO** (GenBio AI, 2024, arXiv:2412.06993) | 模块化多尺度框架 |
| **Omni-DNA** (arXiv:2502.03499) | 多任务基因组基础模型 |

### 实现路径
1. 扩展`DNACharTokenizer`支持RNA字符（A/U/C/G）和蛋白质氨基酸（20种）
2. 预训练数据增加RNA序列（如RefSeq mRNA）
3. 增加跨模态对齐损失：`L_align = contrastive_loss(DNA_latent, RNA_latent)` for paired DNA-RNA sequences
4. Source Matrix S提供天然的sequence-to-token对齐，可直接用于CDS-to-codon的映射分析

### 验证方案
- **中心法则任务**：基因表达预测（DNA→RNA expression）、蛋白质适应度预测（DNA→protein fitness）
- **Tokenization可解释性**：分析CDS区域学到的token边界是否对应密码子边界
- **Transfer能力**：在DNA上预训练的model，zero-shot迁移到RNA任务的表现

### 创新性评估
- **新颖度**：高。"可学习动态tokenizer统一中心法则"之前没有人做过
- **实现难度**：高。需要多模态数据准备和较长的预训练
- **NeurIPS适合度**：很高。跨模态+可学习tokenization的故事很有吸引力

---

## 方向六：Biologically-Interpretable Token Merging（生物可解释的Token合并）

### 故事线
MergeDNA论文（Section 4.4）展示了不同基因组区域的token长度分布不同，但未深入验证token边界是否对应已知的生物学单元。NeurIPS审稿人越来越关注**可解释性**——不仅要好的benchmark数字，还要证明模型学到了生物学上真实的东西。

### 核心Idea
**Interpretable MergeDNA**：
1. 利用Sparse Autoencoders（SAE, 参考decode-gLM）分析MergeDNA的merge embedding空间，发现可解释的生物特征
2. 将学到的token边界与已知功能注释（Ensembl/ENCODE）对齐，量化overlap
3. 设计一个**boundary-aware pretraining objective**：在已知功能边界（exon-intron, promoter-gene）处增加额外的boundary prediction loss
4. Activation steering：通过操控merge embedding控制token粒度，实现可控的序列生成

### 关键论文与资源
| 资源 | 用途 |
|------|------|
| **decode-gLM** (Interpretability for Genomic LMs via SAE) | SAE可解释性方法 |
| `reference_repos/decode-glm/` | 已有decode-gLM参考实现 |
| **ENCODE项目** | 功能注释数据（promoter/enhancer/TF binding sites） |
| **Ensembl注释** | exon/intron/UTR边界 |

### 实现路径
1. 预训练MergeDNA后，收集不同基因组区域的merge boundary位置
2. 与ENCODE/Ensembl功能注释计算overlap统计（Jaccard, precision/recall）
3. 训练SAE on merge embeddings，分析发现的latent features
4. 可选：增加boundary-aware loss `L_boundary = BCE(predicted_boundaries, known_functional_boundaries)`

### 验证方案
- **定量可解释性**：merge boundary vs functional annotation的overlap score
- **Per-context分析**：promoter区域的token粒度 vs enhancer vs intergenic
- **Perturbation实验**：强制改变某些区域的merge粒度，观察下游预测的变化
- **生物学案例研究**：选择几个well-characterized的基因座位(locus)做深入分析

### 创新性评估
- **新颖度**：高。DNA动态tokenizer的可解释性分析是全新方向
- **实现难度**：中等。可解释性分析不需要重新预训练
- **NeurIPS适合度**：很高。"生物学可解释性"是审稿人非常看重的差异化因素

---

## 综合推荐：组合方案

对于一篇NeurIPS论文，我建议以下**组合策略**，选择2-3个方向形成一个coherent story：

### 方案A：效率+长上下文故事（工程导向，最容易复现验证）

> **"MergeDNA-Long: Scaling Context-Aware Genome Modeling to 100K Bases via Hybrid Token Merging"**

组合：**方向二（Hybrid SSM-Attention）+ 方向一（Entropy-Guided Merging）+ 方向三的部分（Learned Compression Schedule）**

- 用entropy-guided merging提升压缩质量 → 更激进的压缩率
- 用hybrid SSM-Attention Latent Encoder处理更长的latent sequence
- 展示在LRB长程基准（20K-40K bp）上的SOTA性能
- 效率分析：FLOPs/Memory vs 序列长度的scaling curve

**需要的额外资源**：
- `pip install flash-linear-attention` (Gated DeltaNet kernels)
- 或 `pip install mamba-ssm` (Mamba-2/3)
- LRB benchmark数据（已在finetune config中配置）

**预估工作量**：4-6周代码实现 + 2-3周实验

---

### 方案B：可解释性+生物学洞察故事（学术价值高，NeurIPS审稿人偏好）

> **"Understanding Learned DNA Tokenization: Interpretable Dynamic Segmentation for Genome Foundation Models"**

组合：**方向六（Interpretable Token Merging）+ 方向一（Entropy-Guided Merging）+ 方向三的部分（DTEM）**

- 系统性分析MergeDNA学到的token边界 vs 已知功能注释
- 引入entropy-guided + DTEM改进合并质量
- 展示改进后的合并策略在benchmark上的提升
- 提供生物学case study证明学到的tokenization有生物学意义

**需要的额外资源**：
- ENCODE功能注释数据
- Ensembl基因注释（exon/intron/UTR边界）
- `reference_repos/decode-glm/` (SAE分析)

**预估工作量**：3-5周代码实现 + 3-4周分析和可视化

---

### 方案C：跨模态故事（最有野心，影响力最大）

> **"Adaptive Central Dogma Modeling via Learnable Cross-Modal Token Merging"**

组合：**方向五（Cross-Modal Dynamic Tokenization）+ 方向二（Hybrid Latent Encoder）**

- 将MergeDNA的动态tokenizer扩展到DNA+RNA的统一建模
- Token Merging在DNA模态学密码子/调控元件边界，在RNA模态学剪接/结构边界
- Hybrid Latent Encoder支持长上下文的跨模态建模
- 在基因表达预测、蛋白质适应度预测等中心法则任务上验证

**需要的额外资源**：
- RNA序列预训练数据（RefSeq mRNA/ncRNA）
- 基因表达数据（GTEx）
- CDBridge/Life-Code论文和数据

**预估工作量**：6-8周代码实现 + 3-4周实验

---

## 实施优先级建议

| 优先级 | 方向 | 理由 |
|--------|------|------|
| 1 | 方向二（Hybrid SSM-Attention） | 效果明确，有HybriDNA验证，直接提升效率和上下文长度 |
| 2 | 方向一（Entropy-Guided Merging） | 实现简单，生物学动机强，与方向二互补 |
| 3 | 方向六（Interpretability） | 不需要重新预训练，NeurIPS审稿人看重 |
| 4 | 方向三（DTEM + Learned Schedule） | 实现简单，可作为方向一的补充 |
| 5 | 方向四（Mixture-of-Depths） | 与方向二有重叠，可二选一 |
| 6 | 方向五（Cross-Modal） | 工作量最大，但影响力最高 |

---

## 需要获取的外部资源清单

### 论文（按优先级）
1. DTEM (NeurIPS 2024): https://arxiv.org/abs/2410.xxxxx (Lee & Hong)
2. BLT (Meta 2024): https://arxiv.org/abs/2412.09871
3. Gated DeltaNet (ICLR 2025): https://arxiv.org/abs/2412.06464
4. HybriDNA (2025): https://arxiv.org/abs/2502.10807
5. Mixture-of-Depths (2024): https://arxiv.org/abs/2404.xxxxx (Raposo et al.)
6. MaMe - Token Merging for SSMs: https://arxiv.org/abs/2508.13599
7. NTv3 (Dec 2025): https://www.biorxiv.org/content/10.64898/2025.12.22.695963v1
8. CDBridge (2025): https://openreview.net/forum?id=Hk4Fb6kaYF
9. Life-Code (2025): https://arxiv.org/abs/2502.07299

### 代码库
1. `flash-linear-attention`: https://github.com/fla-org/flash-linear-attention (GLA/DeltaNet/HGRN2高效实现)
2. `mamba`: https://github.com/state-spaces/mamba (Mamba-2/3)
3. 已有 `reference_repos/blt/`, `reference_repos/hnet/`, `reference_repos/Mixture-of-depths/`, `reference_repos/mixture_of_recursions/`, `reference_repos/ToR_SSM/`, `reference_repos/decode-glm/`

### 数据
1. Multi-Species Genomes（预训练，已配置）
2. ENCODE功能注释：https://www.encodeproject.org/
3. LRB Benchmark：https://github.com/Trop-LongRange/genomics-long-range-benchmark
4. SpliceAI数据：https://github.com/Illumina/SpliceAI
5. GTEx基因表达数据（用于LRB eQTL/Bulk RNA tasks）
