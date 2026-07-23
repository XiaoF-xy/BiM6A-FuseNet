# BiM6A-FuseNet

这是可独立复制到服务器运行的 m6A 位点预测项目。所有版本都采用与 MKE-ResNet 可比较的严格五折模型选择流程，并在固定独立测试集上进行五模型 soft voting。

## 版本

- `v0a_birna_nuc_lora`：纯 BiRNA-BERT NUC 单分支；masked mean pooling，Wqkv LoRA，不含 FiLM 和手工特征。
- `v0b_birna_nuc_fullft`：与 v0a 使用相同的单分支分类头，完整微调 BiRNA-BERT，并使用 10% warmup 后恒定学习率。
- `v0c_birna_lora_full_attention`：保持 v0a 训练设置，LoRA 覆盖 `Wqkv` 和注意力输出投影。
- `v0d_birna_lora_attention_ffn`：保持 v0a 训练设置，LoRA 覆盖完整注意力和 FFN 线性层。
- `v0e_birna_nuc_loraplus`：保持 v0a 的 Wqkv LoRA、masked mean 和分类头，只对 LoRA A、LoRA B、分类头使用非对称学习率 `5e-5 / 8e-4 / 1e-4`。
- `v1_baseline`：原 v9a 结构，BiRNA 特征和 128 维手工特征直接拼接。
- `v1b_proj256_concat`：BiRNA 分支和手工特征分支分别经过 `Linear → LayerNorm → GELU → Dropout(0.2)` 投影到 256 维，拼接成 512 维后分类。特征提取器和评估协议不变。
- `v2a_mke_res_eca_native`：四路 ResNet-ECA 手工分支，使用非对称 `1536+128` 融合。
- `v2b_mke_res_eca_proj256`：与 v2a 相同的手工分支，两条分支分别投影到 256 维。
- `v2c_mke_handcrafted_only_official4c`：不含 BiRNA-BERT 的纯手工对照；严格复现官方源码实际使用的 ONEHOT(4)+CHEM(4)+EIIP(1)+ENAC(4) 输入和四路 ResNet-ECA 分类器。
- `v3a_full_mke_eca_native`：在 v2a 的融合后特征图上增加完整 MKE-ECA。
- `v3b_full_mke_eca_proj256`：在 v2b 的融合后特征图上增加完整 MKE-ECA，保留双分支 256 维对齐。
- `v4a_oof_weighted_late_fusion`：不重新训练；用 benchmark OOF 预测按 ACC 学习 `v2c + v0a` 的单一概率权重。
- `v4b_oof_logistic_stacking`：不重新训练；用 benchmark OOF 预测训练正则化逻辑回归，融合 `v2c + v0a`。

## 数据含义

```text
data/m6a_41nt/<dataset>/benchmark.csv         # 原 train.csv，仅用于五折训练/验证
data/m6a_41nt/<dataset>/independent_test.csv  # 原 test.csv，不参与模型选择
data/m6a_41nt/manifest.csv                    # 数据集名称、路径和样本数元数据
```

支持别名：`H_b H_k H_l M_b M_h M_k M_l M_t R_b R_k R_l`。数据内容未清洗或去重，以保持与 MKE/BiRNA_m6A 的可比性；继承的重复和跨集合重叠会写入 `data_audit.json`。

## 运行

先按 [服务器环境说明](env/README_env.md) 创建环境，然后检查命令：

```bash
python train.py --version v1_baseline --dataset H_b --seed 42 --dry_run
```

正式训练：

```bash
python train.py --version v1_baseline --dataset H_b --seed 42
```

运行 v1b 人脑数据集：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v1b_proj256_concat --dataset H_b --seed 42
```

检查四个 MKE 版本命令：

```bash
python train.py --version v2a_mke_res_eca_native --dataset H_b --seed 42 --dry_run
python train.py --version v2b_mke_res_eca_proj256 --dataset H_b --seed 42 --dry_run
python train.py --version v3a_full_mke_eca_native --dataset H_b --seed 42 --dry_run
python train.py --version v3b_full_mke_eca_proj256 --dataset H_b --seed 42 --dry_run
```

运行纯手工官方源码对照：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v2c_mke_handcrafted_only_official4c --dataset H_b --seed 42
```

同时运行三个组织的纯 BiRNA-BERT LoRA 单分支：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0a_birna_nuc_lora --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v0a_birna_nuc_lora --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v0a_birna_nuc_lora --dataset H_l --seed 42
```

完整微调版本使用：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0b_birna_nuc_fullft --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v0b_birna_nuc_fullft --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v0b_birna_nuc_fullft --dataset H_l --seed 42
```

v0b 会训练完整 BiRNA-BERT，显存占用明显高于 v0a；发生 OOM 时应显式调整实验配置并记录新版本，程序不会自动静默减小 batch size。

同时运行三个组织的 v0c 完整 Attention LoRA：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0c_birna_lora_full_attention --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v0c_birna_lora_full_attention --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v0c_birna_lora_full_attention --dataset H_l --seed 42
```

同时运行三个组织的 v0d Attention+FFN LoRA：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0d_birna_lora_attention_ffn --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v0d_birna_lora_attention_ffn --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v0d_birna_lora_attention_ffn --dataset H_l --seed 42
```

同时运行三个组织的 v0e Wqkv LoRA+：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0e_birna_nuc_loraplus --dataset H_b --seed 42
CUDA_VISIBLE_DEVICES=1 python train.py --version v0e_birna_nuc_loraplus --dataset H_k --seed 42
CUDA_VISIBLE_DEVICES=2 python train.py --version v0e_birna_nuc_loraplus --dataset H_l --seed 42
```

v0e 的模型结构与 v0a 相同，只改变优化器参数组学习率。是否提升应以 benchmark 五折汇总结果判断，独立测试集不参与版本选择。

v2c 不加载 tokenizer 或 BiRNA-BERT 权重。它使用 batch size 64、Adam、初始学习率 `1e-3`、weight decay `1e-5`、最多 100 轮、验证损失调度耐心 10 和验证 ACC 早停耐心 20。模型输出两个 logits，因此采用与公开源码接口一致的交叉熵；论文方法部分写作 BCE，这一出处差异记录在 `docs/superpowers/specs/2026-07-18-v2c-official-handcrafted-design.md`。

运行 v4a 和 v4b 前，必须先完成同一数据集、同一 seed 的 `v2c_mke_handcrafted_only_official4c` 和 `v0a_birna_nuc_lora`。v4 只读取已有 CSV，不使用 GPU：

```bash
python train.py --version v4a_oof_weighted_late_fusion --dataset H_b --seed 42
python train.py --version v4b_oof_logistic_stacking --dataset H_b --seed 42
```

如果基础结果位于项目外的目录，例如 `../outputs`：

```bash
python train.py --version v4a_oof_weighted_late_fusion --dataset H_b --seed 42 --outputs_root ../outputs
python train.py --version v4b_oof_logistic_stacking --dataset H_b --seed 42 --outputs_root ../outputs
```

v4 的 benchmark 指标使用按折留出的第二层交叉拟合：每次只用另外四折 OOF 概率拟合融合规则，再预测当前折。最终融合规则使用全部 benchmark OOF 概率拟合，然后只在独立测试集上评估一次；独立集标签不参与权重、元模型或阈值选择。

`v4c_oof_weighted_threshold_tuned` 是 v4a 的阈值调优对照：它在每个 benchmark meta 训练集上联合搜索手工分支权重 `alpha=0.00...1.00` 和分类阈值 `threshold=0.30...0.70`（步长均为 0.01），以固定的 ACC 为唯一选择指标。它不重新训练基础模型、不使用 GPU；独立测试集标签不参与 `alpha`、阈值或版本选择。

```bash
python train.py --version v4c_oof_weighted_threshold_tuned --dataset H_b --seed 42
```

五折固定为 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。每折训练 seed 依次为 42–46，按验证 ACC 选择最佳 epoch。独立测试集最终结果是五个折模型正类概率的平均值，不是五次测试指标的平均值。

所有版本共享训练加载规则：训练集不足一个完整 batch 的尾批会被丢弃，以避免 BatchNorm 在单样本 batch 上无法计算统计量；验证集、benchmark OOF 和独立测试集始终保留全部样本。

## 输出

结果保存在 `outputs/<version>/<dataset>/seed_<seed>/`：

```text
benchmark_cv_metrics.csv
benchmark_cv_summary.json
independent_ensemble_metrics.json
independent_ensemble_predictions.csv
data_audit.json
fold_01/ ... fold_05/
  train_log.csv
  benchmark_predictions.csv
  independent_predictions.csv
  metrics.json
plots/
  independent_roc_pr.png
  independent_roc_pr.pdf
  independent_roc_curve.csv
  independent_pr_curve.csv
```

v4 结果目录不包含神经网络训练日志和折模型，额外保存 `benchmark_meta_oof_predictions.csv` 与 `fusion_model.json`；其余独立集指标、预测和 ROC/PR 文件命名与已有版本一致。

这些 CSV 可重新绘制或与其他模型叠加绘图。折模型只作为训练过程中的临时文件：两个预测文件和指标成功写入并可回读后立即删除，以减少服务器磁盘占用。预训练 BiRNA-BERT 权重不会删除。

## 迁移到服务器

先在本机确认目录完整：

```bash
python scripts/verify_portable.py
```

请复制整个 `BiM6A-FuseNet` 文件夹，包括 `pretrained/birna-bert-model/pytorch_model.bin`。为了避免把本地 Git 历史和输出一起传输，推荐：

```bash
rsync -av --exclude .git --exclude outputs BiM6A-FuseNet/ user@server:/path/BiM6A-FuseNet/
```

不要仅使用普通 `git clone` 部署：447 MB 权重超过常见 Git 托管的单文件限制，因此本地 Git 忽略该文件；`rsync`、`scp` 或 Finder 复制完整文件夹才会带上它。权重 SHA-256 为：

```text
4833ca3207d1908a86acffc84d6435379ab65c8da8f1790065c3c683bdacef3b
```

项目运行时不依赖旁边的 `BiRNA_m6A` 或 `MKE-Resnet` 文件夹。
