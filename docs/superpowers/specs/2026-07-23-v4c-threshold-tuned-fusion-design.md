# v4c OOF Threshold-Tuned Weighted Fusion Design

## Goal

Add `v4c_oof_weighted_threshold_tuned`, a no-GPU fusion experiment that jointly selects the handcrafted probability weight and binary decision threshold using benchmark OOF ACC only.

## Inputs and isolation

v4c reads the same completed base outputs as v4a: `v2c_mke_handcrafted_only_official4c` and `v0a_birna_nuc_lora`. It aligns rows by sample ID, sequence, and label. Independent-test labels are never used to fit a weight, choose a threshold, or choose a version.

## Search rule

For every pair in `alpha ∈ {0.00, ..., 1.00}` and `threshold ∈ {0.30, ..., 0.70}`, calculate:

`p = alpha * p_handcrafted + (1 - alpha) * p_birna`

and maximize ACC for predictions `p >= threshold`.

ACC ties are resolved deterministically by: threshold closest to 0.50, alpha closest to 0.50, smaller alpha, then smaller threshold.

## Evaluation protocol

Benchmark reporting uses the existing five-fold, fold-preserving meta cross-fit. For each held-out base OOF fold, v4c fits `(alpha, threshold)` on the other four folds and predicts only the held-out fold. The final independent-test rule is fitted once on all benchmark OOF rows and applied to the two aligned independent five-model soft-voting probabilities.

## Compatibility

v4a remains fixed-threshold weighted fusion and v4b remains logistic stacking. v4c writes the same artifact set, with `fusion_model.json` and `resolved_config.json` recording the selected threshold and ACC selection metric.
