# MergeDNA 本地复现实验结果对照论文

参考论文：

- `papers/reference_papers/MergeDNA Context-aware Genome Modeling with Dynamic Tokenization.pdf`

## 1. 本次运行实际覆盖了什么

你执行的命令是：

```bash
bash scripts/finetune_local.sh --genomic_benchmark_data_dir ~/.genomic_benchmarks
```

结合代码可以确认：

- `scripts/finetune_local.sh` 默认会走 `finetune_all_gb`，也就是依次跑完 8 个 Genomic Benchmark 任务，而不是 NT / GUE / SpliceAI / LRB / Protein 等实验。
- 当前本地微调配置来自 `configs/finetune_local.yaml`。
- 这套本地配置加载的是 `./outputs/pretrain_local/checkpoint-20000.pt`，不是论文里的 100K step 完整预训练 checkpoint。

也就是说，这次结果应被理解为：

- `本地单卡 + 20K steps 预训练 checkpoint + Genomic Benchmark 8 任务` 的一次复现实验；
- 不是论文全设定的完整复现。

## 2. 当前结果是否合理

### 2.1 总体判断

结论：**整体合理，但明显低于论文完整版结果，且当前评估流程有一个会“偏乐观”的 caveat。**

理由如下：

1. 8 个任务全部正常产出，没有报错，说明流程本身是通的。
2. `accuracy`、`f1` 很接近，这和 Genomic Benchmark 的平衡二分类设定是吻合的。
3. `mcc` 随着 `accuracy` 变化基本同步，没有出现“accuracy 很高但 mcc 很低”的异常情况。
4. 只有 `human_ensembl_regulatory` 一项略高于论文，其他 7 项都低于论文，整体趋势符合“本地轻量复现 < 论文完整训练”的预期。

### 2.2 为什么低于论文是正常的

主要原因不是代码明显跑错，而是实验设置本身和论文不同：

1. 论文 Appendix A1 使用的是 `100K` 预训练 steps；当前本地配置只用了 `20K` steps 的 `pretrain_local` checkpoint。
2. 论文结果是 `3 次运行取平均`；你这次是单次本地运行。
3. 论文 Appendix A2 / Table 1 对 Genomic Benchmark 报的是正式复现实验；你现在跑的是本地单卡版配置，batch size、可用算力、超参搜索强度都更弱。
4. Genomic Benchmark 每条序列本身只有 `200bp`，所以这里的差距**主要不是** `finetune_local.yaml` 里 `max_seq_length=1024` 导致的截断问题；更可能是预训练强度不足和单次运行方差造成的。

### 2.3 当前代码里的一个重要 caveat

当前训练代码在每个 epoch 之后，直接用 `test` split 做评估，并按这个结果保存“best model”：

- `train.py` 中，Genomic Benchmark 直接把 `test` split 传给了 `runner.train(..., test_loader)`。
- `mergedna/training/finetune.py` 中，每个 epoch 都会在这个 loader 上评估，并按最佳指标保存 `best_model.pt`。

这意味着当前日志里的 “Best results” 其实是：

- `在 test set 上按 epoch 反复查看后挑出来的最好值`

它通常会比“严格只在最终 test 上评一次”的结果更乐观一些。换句话说：

- **即便在这种偏乐观的评估口径下，当前结果仍整体低于论文，说明复现差距是真实存在的。**

## 3. 已有实验结果 vs 论文

说明：

- 论文对 Genomic Benchmark 的逐任务结果来自 Appendix `Table A2`。
- 你当前日志里的 `Average Accuracy = 0.8346` 是 `8 个子任务的直接平均`。
- 论文正文 `Table 1` 的 `90.87` 是按 `Enhancers / Species Classification / Regulatory Elements` 三大组先求组均值、再取平均，因此和 Appendix A2 的“直接 8 任务平均”不是同一个口径。

### 3.1 逐任务对比

| 本地任务名 | 论文表中的任务名 | Train | Test | 当前 acc (%) | 论文 acc (%) | 差值 (pp) | 当前 MCC | 当前 F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `human_enhancers_cohn` | Human Enhancers Cohn | 20843 | 6948 | 72.61 | 76.54 | -3.93 | 0.4525 | 0.7260 |
| `human_enhancers_ensembl` | Human Enhancers Ensembl | 123872 | 30970 | 82.54 | 93.18 | -10.64 | 0.6513 | 0.8253 |
| `demo_coding_vs_intergenomic_seqs` | Coding vs Intergenomic | 75000 | 25000 | 89.61 | 96.02 | -6.41 | 0.7923 | 0.8961 |
| `demo_human_or_worm` | Human vs Worm | 75000 | 25000 | 95.22 | 97.65 | -2.43 | 0.9045 | 0.9522 |
| `dummy_mouse_enhancers_ensembl` | Mouse Enhancers | 968 | 242 | 76.45 | 85.62 | -9.17 | 0.5298 | 0.7643 |
| `human_ensembl_regulatory` | Human Regulatory | 231348 | 57713 | 93.76 | 93.49 | +0.27 | 0.9062 | 0.9394 |
| `human_nontata_promoters` | Human NonTATA Promoters | 27097 | 9034 | 84.79 | 96.34 | -11.55 | 0.7029 | 0.8478 |
| `human_ocr_ensembl` | Human OCR Ensembl | 139804 | 34952 | 72.67 | 82.16 | -9.49 | 0.4538 | 0.7267 |

### 3.2 分组对比

#### 按论文 Table 1 的分组口径

| 分组 | 当前平均 acc (%) | 论文平均 acc (%) | 差值 (pp) |
|---|---:|---:|---:|
| Enhancers (3) | 77.20 | 85.11 | -7.91 |
| Species Classification (2) | 92.42 | 96.84 | -4.42 |
| Regulatory Elements (3) | 83.74 | 90.66 | -6.92 |
| Table 1 式组均值平均 | 84.45 | 90.87 | -6.42 |

#### 按当前日志的直接 8 任务平均口径

| 指标 | 当前 | 论文 | 差值 |
|---|---:|---:|---:|
| 8-task direct mean accuracy (%) | 83.46 | 90.12 | -6.67 |

### 3.3 哪些结果最接近论文

- 最接近论文的是 `human_ensembl_regulatory`，当前甚至高出 `+0.27pp`。
- 次接近的是 `demo_human_or_worm`，只低了 `-2.43pp`。
- 掉点最明显的是：
  - `human_nontata_promoters`: `-11.55pp`
  - `human_enhancers_ensembl`: `-10.64pp`
  - `human_ocr_ensembl`: `-9.49pp`
  - `dummy_mouse_enhancers_ensembl`: `-9.17pp`

### 3.4 对这些差距的解释

从结果形态看，当前模型对“较容易的物种分类”保留了不错的能力，但对 enhancer / promoter / OCR 这类更依赖稳健表征质量的任务掉点明显，这与“预训练还不够充分”是相符的。

因此，如果只问一句“这次跑出来的结果像不像真的、合不合理”，我的判断是：

- **像真的，且和当前本地配置是匹配的。**
- **它更像是一个有效但偏早期的 local reproduction，而不是论文最终性能。**

## 4. 运行时长

### 4.1 说明

当前仓库没有把每个任务的开始/结束时间结构化写入日志文件，因此无法 100% 精确恢复真实耗时。

下面的时间是基于这些文件时间做的**粗粒度估计**：

- 各任务输出目录在第一次保存 `best_model.pt` 时被创建；
- 我用相邻任务输出目录的创建时间差，近似当作该任务的 wall-clock slot；
- 最后一个任务用其目录创建时间到 `genomic_benchmark_results.json` 写出时间的差值估计；
- 因此每个任务可能存在约 `1 个 epoch` 量级的误差，但总时间量级基本可信。

### 4.2 粗粒度时间线

| 任务 | 估计 wall-clock slot | Train samples | 每 epoch step 数（batch=16） |
|---|---:|---:|---:|
| `human_enhancers_cohn` | 03:28:09 | 20843 | 1303 |
| `human_enhancers_ensembl` | 12:33:09 | 123872 | 7742 |
| `demo_coding_vs_intergenomic_seqs` | 08:12:59 | 75000 | 4688 |
| `demo_human_or_worm` | 07:40:15 | 75000 | 4688 |
| `dummy_mouse_enhancers_ensembl` | 02:30:50 | 968 | 61 |
| `human_ensembl_regulatory` | 22:40:42 | 231348 | 14460 |
| `human_nontata_promoters` | 04:14:42 | 27097 | 1694 |
| `human_ocr_ensembl` | 13:30:39 | 139804 | 8738 |
| **总时长（粗估）** | **74:51:24** | - | - |

补充观察：

- 最慢的几个任务恰好也是训练集最大的几个：`human_ensembl_regulatory`、`human_ocr_ensembl`、`human_enhancers_ensembl`。
- `dummy_mouse_enhancers_ensembl` 数据量极小，所以很快是合理的。

## 5. 论文里有、但这次没有运行的实验

本次命令默认只会跑 `finetune_all_gb`，因此下面这些论文结果**当前都还没有补齐**。

| 论文实验 | 论文结果 | 当前状态 | 备注 |
|---|---|---|---|
| NT Benchmark (18 tasks), Table 2 | 平均 `78.39` | 未运行 | 仓库已有 `finetune_all_nt` 入口，可后续补齐 |
| GUE Benchmark (24 tasks), Table 3 | 平均 `77.11` | 未运行 | 仓库已有 `finetune_all_gue` 入口，但数据仍需手动下载 |
| SpliceAI, Table 4 | Donor `64.4`，Acceptor `74.5`，Mean AUROC `69.8` | 未运行 | 已补 `scripts/run_all_experiments.py` + `mergedna/experiments/spliceai.py`，待下载原始资源并执行 |
| Protein Fitness, Table 5 | Bacteria SRCC `42.72`，Human SRCC `20.58` | 未运行 | 已补 `mergedna/experiments/protein_fitness.py`，但若没有 assay 级 DNA 变体，当前实现会使用近似 codon 映射 |
| LRB, Table 6 | Causal eQTL AUROC `0.75`，Bulk RNA `R^2=0.62` | 未运行 | 已补 `mergedna/experiments/lrb.py`，待下载 `hg19/hg38 + LRB csv` 后执行 |
| Ablation, Table 7 | 选中最佳设置相对 byte-tokenizer baseline `+1.57` | 未运行 | 已补 `mergedna/experiments/ablation.py`，会先各自预训练再跑 GB |

### 5.1 NT Benchmark：后续建议补齐

论文 Table 2 的 headline 是：

- `Average (18 tasks) = 78.39`

如果后续补齐，建议优先跑这个，因为：

- 仓库已经有 `finetune_all_nt` 模式；
- `scripts/finetune_nt.sh` 也已经列好了 18 个任务；
- 和当前 Genomic Benchmark 一样，都属于现有代码路径能直接支持的范围。

### 5.2 GUE Benchmark：后续建议补齐

论文 Table 3 的 headline 是：

- `Average (24 tasks) = 77.11`

当前仓库也已经支持：

- `train.py --mode finetune_all_gue`

但要注意：

- `scripts/download_data.py` 里明确写了 GUE 需要手动从 Google Drive 下载并解压。

### 5.3 Multi-omics / 长序列 / 消融：现在已经补了统一入口，但结果仍待运行

论文还报告了：

- SpliceAI（Table 4）
- Protein Fitness（Table 5）
- LRB（Table 6）
- Ablation（Table 7）

这些实验在论文里很重要。现在仓库里已经新增了统一总入口：

- `scripts/run_all_experiments.py`
- `scripts/run_all_local.sh`
- `scripts/run_all_a800.sh`

并且支持：

- `--groups` 选择实验子集
- `--skip-existing` 跳过已有结果
- `--dry-run` 检查缺失资源
- `--prepare-only` 先做数据准备

但要注意，**当前只是把实验配置和运行链路补齐了，还没有实际把这些缺失实验跑完**。

## 6. 追加：预训练部分

## 6.1 你之前跑到的预训练步数

从脚本和配置看，如果你执行的是默认命令：

```bash
bash scripts/pretrain_local.sh
```

那么它会默认读取 `configs/pretrain_local.yaml`，而这份配置明确写的是：

- `max_steps: 20000`
- `batch_size: 4`
- `gradient_accumulation: 4`
- `max_seq_length: 1024`
- `output_dir: ./outputs/pretrain_local`

所以你的判断是对的：

- **当前本地预训练确实不是论文的 100K steps，而是 20K steps 本地版。**

## 6.2 当前本地预训练 vs 论文预训练

### 6.2.1 设定对比

| 项目 | 当前本地预训练 | 论文 Appendix A1 |
|---|---|---|
| 模型规模 | 380M | 380M |
| Local / Latent / Decoder 层数 | 4 / 20 / 4 / 2 | 4 / 20 / 4 / 2 |
| 最大序列长度 | 1024 | 4096 |
| 训练步数 | 20000 | 100000 |
| 每卡 batch | 4 | 8 |
| 梯度累积 | 4 | 16 |
| 有效 batch | 16 | 256 |
| 学习率 | 1e-4 | 1e-4 |
| warmup | 1000 | 10000 |
| 硬件 | 单卡 RTX 5070 Ti 16GB | 8 × A100-80G |
| 训练时长 | 本地实测约 15 小时量级 | 论文写明接近 5 天 |

### 6.2.2 训练规模差距有多大

如果只按“样本条数”粗略估算：

- 当前本地版处理的总样本量约为 `20000 × 16 = 320,000`
- 论文版约为 `100000 × 256 = 25,600,000`

也就是：

- **当前本地版只有论文预训练样本量的约 `1.25%`**

如果进一步考虑序列长度：

- 当前本地 token budget 约为 `20000 × 16 × 1024 = 327,680,000`
- 论文 token budget 约为 `100000 × 256 × 4096 = 104,857,600,000`

也就是：

- **当前本地版只有论文 token budget 的约 `0.31%`**

这个比例已经足以解释为什么后面的微调结果整体低于论文。

### 6.2.3 一个需要特别注意的配置差异

仓库里还有 `configs/pretrain.yaml` 和 `configs/pretrain_a800.yaml`，其中注释把它们描述成“论文复现”或“匹配论文”。

但对照论文 Appendix A1，可以看到至少这些关键项并不完全一致：

- 论文写的是总 batch `256`，而仓库 full pretrain 配置写的是有效 batch `32`
- 论文写的是 warmup `10000`，而仓库 full pretrain 配置写的是 `2000`
- 论文写的是 `8 × A100-80G`，而仓库 `pretrain_a800.yaml` 是单卡 A800 近似版

所以如果你后续要做“严格论文复现”，建议：

- **以论文 Appendix A1 为主，再反向校正 repo 里的 full-pretrain 配置。**

## 6.3 当前预训练产物情况

`outputs/pretrain_local` 里目前能看到这些 checkpoint：

- `checkpoint-100.pt`
- `checkpoint-2000.pt`
- `checkpoint-4000.pt`
- `checkpoint-6000.pt`
- `checkpoint-8000.pt`
- `checkpoint-10000.pt`
- `checkpoint-12000.pt`
- `checkpoint-14000.pt`
- `checkpoint-16000.pt`
- `checkpoint-18000.pt`
- `checkpoint-20000.pt`

这说明：

- 至少有一次很早期的短跑，产出了 `checkpoint-100.pt`
- 之后又完成了一次完整主跑，最终到达了 `checkpoint-20000.pt`

同时，当前仓库并没有把预训练 loss 曲线结构化写到本地日志文件里，因此现在能可靠恢复的是：

- `checkpoint 时间线`
- `预训练实际跑到的步数`

但不能像微调那样，直接从本地文件里回溯完整 loss 曲线与最优点。

## 6.4 预训练运行时长

### 6.4.1 可直接确认的时间线

从 checkpoint 文件时间可以直接确认：

| 事件 | 时间 |
|---|---|
| `checkpoint-100.pt` 创建 | 2026-03-25 05:32:15 |
| `checkpoint-2000.pt` 创建 | 2026-04-02 15:06:00 |
| `checkpoint-20000.pt` 创建 | 2026-04-03 04:50:38 |

这说明本地预训练不是一次连续跑完的单次作业，而更像是：

1. 先做了一次很短的早期试跑
2. 后来重新启动并续跑到 20K

### 6.4.2 主跑阶段的可见时长

从 `checkpoint-2000.pt` 到 `checkpoint-20000.pt` 的可见主跑时间是：

- **13:44:37**

对应该区间的平均速度约为：

- **2.75 s/step**

### 6.4.3 估计的完整主跑时长

由于第一次可见 checkpoint 已经是 `2000 step`，前 0~2000 step 的精确时间无法直接从文件恢复。

如果用主跑阶段平均速度 `2.75 s/step` 回推，那么：

- 若按 `0 -> 20000` 估计，总时长约 **15:16:15**
- 若考虑这次大概率是从 `checkpoint-100` 自动续跑，则 `100 -> 20000` 的主跑总时长约 **15:11:40**

因此比较稳妥的结论是：

- **这次 20K 本地预训练主跑总时长约 15 小时量级。**

### 6.4.4 为什么比配置注释里的 11 小时更长

`configs/pretrain_local.yaml` 里写的是：

- `bs=4 accum=4 -> 1.92s/step, 20K steps ≈ 11h`

但当前文件时间线显示，实际 wall-clock 更接近 15 小时量级。原因大概率有两类：

1. 实际机器状态与当时 benchmark 注释时不同
2. 中间存在明显的暂停或降速

从 checkpoint 区间看：

- `2000 -> 14000` 基本稳定在约 `2.08 s/step`
- 但 `14000 -> 18000` 被拉长到了约 `4.94 s/step`

因此这段时间线上很像出现过：

- 训练暂停
- 机器负载变化
- I/O 或显存相关的降速

所以对外记录时，建议写成：

- `可见主跑时长 13h44m`
- `完整主跑估计约 15h12m`
- `比配置注释中的 11h 更长，且中间存在疑似暂停/降速`

## 6.5 预训练部分的结论

如果只看“项目是否真的完成了本地预训练”：

- 是的，已经产出了直到 `checkpoint-20000.pt` 的完整本地预训练结果。

如果看“它和论文预训练是不是一回事”：

- 不是。

更准确地说，它是：

- **论文同架构的本地缩小版预训练**

而不是：

- **论文同规模、同 token budget、同硬件条件的正式复现**

这也正是当前下游 Genomic Benchmark 分数明显低于论文的最核心原因之一。

## 7. 最终结论

如果把这次结果放在“本地单卡、20K-step 预训练 checkpoint、只跑 Genomic Benchmark”的前提下看：

- 结果是**合理的**；
- 说明当前实现**基本是通的**；
- 但与论文最终结果仍有明显差距；
- 这个差距更像是“训练规模 / checkpoint 强度 / 评估 protocol / 单次运行方差”的复现差距，而不是明显的程序错误。

如果你的目标是逐步逼近论文结果，下一步建议优先级如下：

1. 先把预训练配置尽量向论文 Appendix A1 对齐
2. 用更接近论文的预训练 checkpoint 重新跑 `Genomic Benchmark`
3. 再补 `NT Benchmark`
4. 再补 `GUE Benchmark`
5. 最后再补 `SpliceAI / LRB / Protein / Ablation`

其中最关键的性能瓶颈，依然大概率在：

- `pretrain_local/checkpoint-20000.pt` 和论文 100K-step 正式预训练模型之间的质量差距。
