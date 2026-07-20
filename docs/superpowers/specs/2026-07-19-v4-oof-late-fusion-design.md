# v4 OOF Late-Fusion Design

## Goal

Add two lightweight, reproducible late-fusion experiments without changing or retraining any existing base model:

- `v4a_oof_weighted_late_fusion`: learn one handcrafted-model probability weight from benchmark OOF predictions.
- `v4b_oof_logistic_stacking`: learn a regularized logistic-regression meta-classifier from the two base probabilities.

Both experiments use `v2c_mke_handcrafted_only_official4c` as the handcrafted expert and `v0a_birna_nuc_lora` as the BiRNA expert.

## Protocol

The benchmark predictions from each base model must be genuine five-fold OOF predictions. Base prediction rows are aligned by fold, sample ID, sequence, and label. The independent predictions are the existing five-model soft-voting files and are aligned by sample ID, sequence, and label.

Benchmark fusion performance is estimated with a second, fold-preserving cross-fit:

1. Hold out one benchmark fold.
2. Fit the fusion rule on the other four OOF folds.
3. Predict the held-out fold.
4. Repeat for all five folds and concatenate the meta-OOF predictions.

The final independent-test fusion rule is then fitted once on all benchmark OOF predictions and applied once to the aligned independent ensemble probabilities. Independent labels are never used for fitting, weight selection, threshold selection, or model selection.

## v4a

For handcrafted weight `alpha`, fused probability is:

`p = alpha * p_handcrafted + (1 - alpha) * p_birna`

Search `alpha` from 0.00 through 1.00 in steps of 0.01 and maximize ACC at threshold 0.5. ACC ties are resolved by choosing the value closest to 0.5, then the smaller alpha for deterministic behavior.

## v4b

Fit `sklearn.linear_model.LogisticRegression` with inputs `[p_handcrafted, p_birna]`, L2 regularization, `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, and the experiment seed. The positive-class probability is evaluated with the unchanged threshold 0.5.

## Interface and outputs

Both versions use the existing entry point:

```bash
python train.py --version <version> --dataset H_b --seed 42
```

They require the two completed base-output directories under the selected output root. They write the usual result artifacts under their own version directory, including benchmark fold metrics, benchmark meta-OOF predictions and summary, independent ensemble predictions and metrics, ROC/PR plot files, a resolved configuration, and a JSON description of the fitted final fusion rule. They do not load a GPU, train a neural network, create checkpoints, or delete base results.

## Validation

Tests cover row alignment, v4a weight fitting, fold-preserving meta cross-fitting, v4b fitting, absence of independent-label fitting, output creation, and command dispatch. Existing neural experiment command construction remains unchanged.
