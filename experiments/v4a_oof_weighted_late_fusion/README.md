# v4a OOF Weighted Late Fusion

This version performs no neural-network training. It combines completed predictions from:

- handcrafted expert: `v2c_mke_handcrafted_only_official4c`
- BiRNA expert: `v0a_birna_nuc_lora`

The fused positive-class probability is `alpha * p_handcrafted + (1 - alpha) * p_birna`. Alpha is searched from 0.00 to 1.00 at 0.01 intervals using benchmark OOF ACC. Benchmark reporting uses a fold-preserving meta cross-fit, while the final alpha is fitted on all benchmark OOF rows before one independent-test evaluation.

```bash
python train.py --version v4a_oof_weighted_late_fusion --dataset H_b --seed 42
```

No GPU or retained model checkpoint is required. The two base result directories must already exist under the selected `--outputs_root`.
