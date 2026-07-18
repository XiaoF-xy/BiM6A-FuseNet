# BiRNA-BERT 单分支 v0a/v0b 实验设计

## 1. 目标

建立不包含手工特征、MKE 编码、FiLM 和双分支融合的纯 BiRNA-BERT 对照实验，回答两个问题：

1. 当前 BiRNA-BERT NUC 表征单独用于 m6A 预测时能达到什么水平？
2. `Wqkv` LoRA 与全参数微调相比，哪一种迁移方式更适合 41nt m6A 数据？

本阶段只研究 BiRNA-BERT 分支。池化方式、分类头、数据划分和评价协议保持一致，避免把融合结构的影响带入结论。

## 2. 实验版本

### 2.1 `v0a_birna_nuc_lora`

用途：作为纯 BiRNA-BERT 的低成本 LoRA 基线。

模型设置：

- NUC tokenization；
- BiRNA-BERT 原始参数冻结；
- LoRA 仅作用于 `Wqkv`；
- `r=8`、`alpha=32`、`dropout=0.05`；
- 不使用 BPE、FiLM、中心窗口 CNN 或任何手工特征。

训练设置：

- AdamW；
- batch size 32；
- 初始学习率 `1e-4`；
- weight decay `0.01`；
- 最多 20 epochs；
- 每折按照 benchmark validation ACC 保存最佳 epoch。

### 2.2 `v0b_birna_nuc_fullft`

用途：验证完整微调能否突破 `Wqkv`-only LoRA 的能力上限。

模型设置与 v0a 完全相同，唯一的模型迁移差异是：

- 不使用 LoRA；
- 不冻结 BiRNA-BERT；
- BiRNA-BERT 与分类头全部参与训练。

训练设置参考 BiRNA-BERT 官方公开的相似短序列微调代码：

- AdamW；
- batch size 64；
- 初始学习率 `1e-6`；
- warmup ratio `0.1`，warmup 后保持常数学习率；
- 最多 10 epochs；
- 每折按照 benchmark validation ACC 保存最佳 epoch。

v0a 与 v0b 的训练超参数分别匹配 LoRA 和完整微调所需的学习率尺度，因此本实验比较的是两套完整迁移协议，而不是只比较一个布尔开关。若 v0b 显著更好，后续再用单独消融实验拆分“完整解冻”和“学习率策略”的贡献。

## 3. 共享模型结构

两个版本必须使用相同的下游结构：

```text
41nt RNA sequence
    ↓
NUC tokenizer（核苷酸之间以空格分隔）
    ↓
BiRNA-BERT token embeddings：[B, 41, 768]
    ↓
去除 CLS、SEP 和 padding
    ↓
masked mean pooling：[B, 768]
    ↓
Linear：768 → 256
    ↓
GELU + Dropout(0.2)
    ↓
Linear：256 → 2
    ↓
m6A / non-m6A logits
```

分类损失使用 `CrossEntropyLoss`。模型输出两个 logits，概率由 softmax 的正类概率得到。

本阶段只使用 mean pooling。中心位点拼接、mean+max、局部 CNN 和多层拼接均保留为后续独立消融，不进入 v0a/v0b。

## 4. 数据和评价协议

沿用项目当前 `strict_cv` 协议：

1. 只对 benchmark 数据进行分层五折交叉验证；
2. 每折用四折训练、一折作为 benchmark validation；
3. 每折按照 validation ACC 选择最佳 epoch；
4. 五折 benchmark validation 预测拼成完整 OOF 预测；
5. 五个折模型分别预测 independent test；
6. independent test 使用五模型正类概率平均进行 soft voting；
7. 独立测试集不参与 epoch、超参数、池化方式或版本选择。

每个版本先使用 seed 42。只有具有明显提升的候选版本才追加其他随机种子。

## 5. 输出产物

目录结构继续使用现有规范：

```text
outputs/<version>/<dataset>/seed_<seed>/
├── resolved_config.json
├── data_audit.json
├── benchmark_cv_metrics.csv
├── benchmark_cv_summary.json
├── independent_ensemble_metrics.json
├── independent_ensemble_predictions.csv
├── fold_01/
│   ├── train_log.csv
│   ├── benchmark_predictions.csv
│   ├── independent_predictions.csv
│   └── metrics.json
├── ... fold_02 至 fold_05
└── plots/
```

至少保存以下指标：

- ACC；
- MCC；
- AUC；
- AUPRC；
- F1；
- Precision；
- Recall/Sensitivity；
- Specificity；
- TP、TN、FP、FN。

每折训练完成并生成所有预测与指标后删除最佳模型权重，避免服务器存储持续增长。删除前必须完成当前折的 validation 和 independent test 推理。

## 6. 版本隔离

实现必须满足：

- 不改变 v1、v1b、v2、v3 和 v2c 的默认行为；
- 新增独立实验配置并注册到现有版本入口；
- 纯 BiRNA 版本不得构造或传入 handcrafted feature tensor；
- v0a 的可训练参数只能是 LoRA 参数和分类头；
- v0b 的 BiRNA-BERT 与分类头参数全部可训练；
- warmup 调度仅在显式配置时启用，不能改变已有版本的 scheduler 行为。

## 7. 错误检查

运行前应明确检查：

- NUC tokenization 后每条 41nt 序列恰好包含 41 个内容 token；
- 模型隐藏维度为 768；
- mean pooling 不包含 CLS、SEP 或 padding；
- v0a 确实加载 PEFT，并且 LoRA target `Wqkv` 能匹配到模块；
- v0b 的 trainable parameter count 明显接近 total parameter count；
- 不存在 benchmark 与 independent test 的非预期数据泄漏；
- 全参数微调发生 CUDA OOM 时直接报出建议的 batch size，而不是静默改变实验配置。

## 8. 验证与测试

实现完成后至少执行：

1. 配置注册测试：两个版本均能通过 `train.py --version ...` 解析；
2. 模型前向测试：输入 batch 后输出形状为 `[B, 2]`；
3. token mask 测试：mean pooling 排除特殊 token 与 padding；
4. 参数冻结测试：v0a 只有 LoRA 与分类头可训练；
5. 全量解冻测试：v0b backbone 参数可训练；
6. 数据路径测试：不生成 handcrafted features；
7. 单折 smoke test：短训练能够生成日志、预测和指标文件；
8. 原版本回归测试：已有 v1-v3 配置和启动命令保持不变。

## 9. 结果解释规则

- 如果 v0b 在 benchmark 五折平均和 OOF 指标上稳定优于 v0a，后续融合以 v0b 作为 BiRNA 分支；
- 如果 v0a 与 v0b 接近，优先保留计算和存储成本更低的 v0a；
- 如果两者都明显低于 BiRNA-BERT 论文结果，下一阶段优先检查模型权重、tokenizer、池化与公开训练协议，不立即增加复杂融合模块；
- 不依据已经反复查看的 independent test 指标选择 v0a 或 v0b，版本选择以 benchmark CV/OOF 为准。

## 10. 本阶段不包含的内容

- 手工特征与 MKE 分支；
- FiLM；
- BPE 双视图；
- 中心局部 CNN；
- mean+center 或 mean+max 池化；
- 最后四层解冻；
- 门控融合和晚期融合；
- 阈值搜索；
- 多随机种子正式统计。

这些内容只有在 v0a/v0b 单分支基线完成后，才作为独立实验逐项加入。
