# v0g BiRNA-BERT NUC DoRA

This version changes only v0a's Wqkv adapter type from ordinary LoRA to PEFT DoRA. The final-layer masked mean, classifier, hyperparameters, and strict MKE-comparable evaluation protocol remain unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0g_birna_nuc_dora --dataset H_b --seed 42 &
CUDA_VISIBLE_DEVICES=1 python train.py --version v0g_birna_nuc_dora --dataset H_k --seed 42 &
CUDA_VISIBLE_DEVICES=2 python train.py --version v0g_birna_nuc_dora --dataset H_l --seed 42 &
wait
```

Experiment artifacts are written under `outputs/v0g_birna_nuc_dora/`.
