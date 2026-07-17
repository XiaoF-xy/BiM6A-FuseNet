# BiM6A-FuseNet

这是可独立复制到服务器运行的 m6A 位点预测项目。当前 `v1_baseline` 与 BiRNA_m6A v9a 保持相同模型结构和训练超参数，并采用与 MKE-ResNet 可比较的严格五折模型选择流程。

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

五折固定为 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。每折训练 seed 依次为 42–46，按验证 ACC 选择最佳 epoch。独立测试集最终结果是五个折模型正类概率的平均值，不是五次测试指标的平均值。

## 输出

结果保存在 `outputs/v1_baseline/<dataset>/seed_<seed>/`：

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
