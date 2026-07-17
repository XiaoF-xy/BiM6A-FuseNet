# BiM6A-FuseNet v1 Design

## Objective

Create a standalone `BiM6A-FuseNet` project for server training. Version `v1_baseline` migrates the BiRNA_m6A v9a model without architectural changes, replaces ambiguous train/test dataset naming with benchmark/independent-test naming, and uses the MKE-style evaluation protocol.

## Provenance

- Source project: `BiRNA_m6A`
- Source experiment: `v9a_birna_v7b_handcrafted_multiscale_cnn`
- New experiment name: `v1_baseline`
- Model display name: `BiM6A-FuseNet-v1`
- Architecture modified in v1: no
- Data contents modified in v1: no
- Evaluation protocol modified in v1: yes

The source v9a name appears only in provenance documentation and resolved configuration, not in public experiment or output paths.

## Project and Naming Conventions

Project root:

```text
BiM6A-FuseNet/
```

Python package and filesystem names use lowercase snake_case. Dataset CLI aliases retain the paper abbreviations (`H_b`, `M_t`, and so on). Fold directories use two digits (`fold_01` through `fold_05`).

Key names:

```text
experiments/v1_baseline/
experiments/v1_baseline/config_v1.py
outputs/v1_baseline/<dataset>/seed_<seed>/
```

## Migration Boundary

Migrate only the components required by BiRNA_m6A v9a:

- v9a experiment configuration
- BiRNA-BERT tokenizer and model-loading code
- FiLM model and v9a classifier dependencies
- handcrafted ONEHOT, NCP, EIIP, and ENAC features
- dataset loading and collation
- training engine, metrics, checkpoint, and prediction serialization
- the complete local `pretrained/birna-bert-model` directory, including weights and tokenizer assets

Do not migrate old experiment versions, outputs, logs, bytecode caches, or test-as-validation aliases.

The new project must not depend on the sibling `BiRNA_m6A` directory at runtime. A copied project must remain runnable after transfer to a server by itself.

## Data Layout and Semantics

Use RNA-appropriate `41nt` naming:

```text
data/m6a_41nt/<dataset>/benchmark.csv
data/m6a_41nt/<dataset>/independent_test.csv
```

Mappings:

- `benchmark.csv` is the unchanged content of BiRNA_m6A `train.csv`, corresponding to MKE `Positive.txt` and `Negative.txt`.
- `independent_test.csv` is the unchanged content of BiRNA_m6A `test.csv`, corresponding to MKE `ind_Positive.txt` and `ind_Negative.txt`.

CSV schema:

```csv
sequence,label
ACGT...,1
TGCA...,0
```

Canonical dataset directory names are lowercase snake_case. `Mouse_test` becomes `mouse_testis` because M-t denotes testis.

`manifest.csv` records dataset identifiers, species, tissue, file paths, positive/negative counts, totals, sequence length, and validation status. It is metadata, not a training input.

Data validation checks length 41, alphabet A/C/G/T, center position A, labels 0/1, class counts, duplicate rows, and exact benchmark/independent overlaps. To retain comparability, v1 reports inherited overlaps but does not remove samples.

## v1 Model

`v1_baseline` preserves the v9a architecture and hyperparameters:

- BiRNA-BERT backbone
- NUC global representation
- center-window local representation
- multi-scale CNN kernels 3, 5, and 7
- FiLM conditioning
- LoRA configuration from v9a
- ONEHOT, NCP, EIIP, and ENAC handcrafted features
- v9a feature fusion and classifier

Any future architecture change starts a new version rather than silently changing v1.

## Evaluation Protocol

Only the strict protocol is supported. The new project does not include `test_as_val`.

### Benchmark five-fold cross-validation

For each tissue, split only `benchmark.csv` using stratified five-fold cross-validation:

```text
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

For each fold:

1. Train on four folds.
2. Evaluate every epoch on the held-out benchmark fold.
3. Select and save the epoch with the highest validation ACC, matching the MKE public source behavior.
4. Save validation labels, probabilities, predictions, and metrics for the selected checkpoint.
5. Reload the selected checkpoint and predict `independent_test.csv` exactly once.

Per-fold seeds are 42, 43, 44, 45, and 46 for folds 1-5.

Benchmark results are the mean and sample standard deviation of the five held-out-fold metrics.

### Independent-test soft voting

Align the five fold prediction files by sequence, label, and stable row identifier. Average positive-class probabilities:

```text
ensemble_probability = (p1 + p2 + p3 + p4 + p5) / 5
```

Use threshold 0.5 for the final class. Compute independent-test metrics once from the ensemble probabilities. Do not average the five fold-level metrics as the primary ensemble result.

## Metrics and Plots

Calculate:

- ACC
- MCC
- AUROC
- AUPR/AUPRC
- sensitivity/recall
- specificity
- precision
- F1
- TP, TN, FP, and FN

Create side-by-side independent-test ROC and precision-recall plots from the five-model ensemble probabilities. Save PNG and vector PDF, plus the underlying curve coordinates as CSV.

## Outputs

```text
outputs/v1_baseline/<dataset>/seed_42/
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
│   ├── metrics.json
│   └── best_model.pt
├── fold_02/
├── fold_03/
├── fold_04/
├── fold_05/
└── plots/
    ├── independent_roc_pr.png
    ├── independent_roc_pr.pdf
    ├── independent_roc_curve.csv
    └── independent_pr_curve.csv
```

## Error Handling

Training stops with a clear error when required data, tokenizer files, or pretrained weights are missing; when a dataset violates its manifest; when fold prediction rows cannot be aligned; or when fewer than five valid fold predictions are available for the ensemble. Partial ensembles are not silently accepted.

## Verification

Verification includes:

1. Compare every migrated benchmark and independent-test sample against BiRNA_m6A after normalization.
2. Test deterministic fold membership for seed 42.
3. Unit-test metrics, specificity, curve generation, and ensemble alignment.
4. Run a small end-to-end smoke test.
5. Run one full tissue dataset and verify that all five fold artifacts and final ensemble outputs are present.
6. Confirm the copied project resolves no runtime imports or files from the sibling BiRNA_m6A directory.

## Server Portability

The project includes the complete BiRNA-BERT model assets and dependency/environment specifications. All configuration paths are project-relative or command-line overridable. The project must run after copying only the `BiM6A-FuseNet` directory to a server.
