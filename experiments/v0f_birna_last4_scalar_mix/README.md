# v0f BiRNA-BERT Last-Four-Layer Scalar Mix

This version keeps the v0a Wqkv LoRA and strict MKE-comparable protocol, but learns four softmax-normalized weights over BiRNA-BERT layers 9–12 before masked mean pooling.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --version v0f_birna_last4_scalar_mix --dataset H_b --seed 42 &
CUDA_VISIBLE_DEVICES=1 python train.py --version v0f_birna_last4_scalar_mix --dataset H_k --seed 42 &
CUDA_VISIBLE_DEVICES=2 python train.py --version v0f_birna_last4_scalar_mix --dataset H_l --seed 42 &
wait
```

Experiment artifacts are written under `outputs/v0f_birna_last4_scalar_mix/`.
