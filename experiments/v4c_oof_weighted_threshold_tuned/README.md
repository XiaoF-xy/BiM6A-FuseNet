# v4c OOF Weighted Threshold-Tuned Fusion

v4c does not train a neural network or require a GPU. It combines completed strict-CV predictions from:

- handcrafted expert: `v2c_mke_handcrafted_only_official4c`
- BiRNA expert: `v0a_birna_nuc_lora`

For each benchmark meta-training split, it jointly searches:

```text
alpha ∈ {0.00, 0.01, ..., 1.00}
threshold ∈ {0.30, 0.31, ..., 0.70}
p = alpha * p_handcrafted + (1 - alpha) * p_birna
pred = (p >= threshold)
```

The pair with maximum ACC is selected. Ties are resolved in order by threshold closest to 0.50, alpha closest to 0.50, smaller alpha, and smaller threshold. Benchmark reporting remains fold-preserving meta cross-fit; the final pair is fitted on all benchmark OOF rows and applied once to the independent ensembles. Independent labels are never used to select either parameter.

```bash
python train.py --version v4c_oof_weighted_threshold_tuned --dataset H_b --seed 42
```
