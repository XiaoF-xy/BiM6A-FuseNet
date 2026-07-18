# v0b BiRNA-BERT NUC Full Fine-Tuning

Pure BiRNA-BERT NUC single-branch experiment. It fine-tunes the complete backbone with masked mean pooling and a constant-after-warmup learning-rate schedule.

```bash
python train.py --version v0b_birna_nuc_fullft --dataset H_b --seed 42
```
