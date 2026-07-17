# BiM6A-FuseNet v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a server-portable BiM6A-FuseNet project whose `v1_baseline` model is structurally identical to BiRNA_m6A v9a, uses MKE-style strict five-fold benchmark selection, soft-votes five independent-test predictions, exports paper-ready metrics and ROC/PR figures, and deletes temporary fold checkpoints after successful export.

**Architecture:** Preserve the BiRNA_m6A source module layout and copy the v9a-dependent model files unchanged so model parity can be checked byte-for-byte and by state-dict signature. Replace the version registry and evaluation entrypoint with a single strict `v1_baseline` protocol, rename data by semantic role, and add isolated reporting modules for fold summaries, ensemble predictions, metrics, and plots.

**Tech Stack:** Python 3.12, PyTorch, Transformers-compatible local BiRNA-BERT assets, scikit-learn, pandas, NumPy, Matplotlib, pytest.

---

## File Map

Create or migrate these files:

- `.gitignore`: exclude outputs, caches, and temporary checkpoints while retaining source and data.
- `README.md`: document standalone server setup, dataset aliases, strict protocol, commands, and outputs.
- `requirements.txt`: merge the runtime requirements needed by v9a and plotting/tests.
- `train.py`: stable project entrypoint.
- `configs/configarg.py`: single-version registry, dataset aliases, project-relative paths, and strict five-fold defaults.
- `experiments/v1_baseline/config_v1.py`: v9a-equivalent model and training overrides under the new name.
- `experiments/v1_baseline/README.md`: provenance and protocol documentation.
- `src/dataset_utils.py`: migrated CSV loading.
- `src/handcrafted_features.py`: migrated v9a handcrafted features unchanged.
- `src/model_birna_nuc.py`: migrated BiRNA-BERT loading and NUC classifier support unchanged.
- `src/model_birna_dual_view.py`: migrated dependency used by the training factory unchanged.
- `src/model_birna_film.py`: migrated v9a model architecture unchanged.
- `src/training_utils.py`: migrated training/evaluation helpers, prediction export, and explicit model-memory cleanup.
- `src/metrics_utils.py`: migrated metrics plus specificity and confusion counts.
- `src/reporting.py`: benchmark fold summaries, five-fold probability ensemble, CSV/JSON export, and checkpoint-safe validation.
- `src/plotting.py`: ROC/PR coordinate and PNG/PDF generation.
- `src/train_cv.py`: strict five-fold orchestration only.
- `scripts/prepare_data.py`: semantic dataset migration and manifest generation.
- `scripts/audit_data.py`: validation and overlap reporting.
- `data/m6a_41nt/manifest.csv`: generated dataset inventory.
- `data/m6a_41nt/<dataset>/benchmark.csv`: migrated BiRNA_m6A train data.
- `data/m6a_41nt/<dataset>/independent_test.csv`: migrated BiRNA_m6A test data.
- `pretrained/birna-bert-model/*`: complete copied model/tokenizer assets.
- `tests/test_model_parity.py`: v1/v9a architecture parity.
- `tests/test_data_migration.py`: exact data and manifest parity.
- `tests/test_metrics.py`: metric and specificity correctness.
- `tests/test_reporting.py`: fold summary, soft voting, alignment, and checkpoint deletion safety.
- `tests/test_cv_protocol.py`: strict fold construction and independent-test isolation.
- `tests/test_plotting.py`: ROC/PR artifacts and curve values.

### Task 1: Migrate the standalone source skeleton and full BiRNA-BERT assets

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `requirements.txt`
- Create: `train.py`
- Create: `configs/__init__.py`
- Create: `experiments/__init__.py`
- Create: `experiments/v1_baseline/__init__.py`
- Create: `src/__init__.py`
- Copy: `pretrained/birna-bert-model/*`

- [ ] **Step 1: Record source file inventory and SHA-256 for the pretrained weight**

Run:

```bash
find ../BiRNA_m6A/pretrained/birna-bert-model -maxdepth 1 -type f -print | sort
shasum -a 256 ../BiRNA_m6A/pretrained/birna-bert-model/pytorch_model.bin
```

Expected: the inventory includes `pytorch_model.bin`, tokenizer files, config files, and custom BERT Python modules; the weight hash is non-empty.

- [ ] **Step 2: Copy only v9a runtime modules and all pretrained assets**

Copy the required source files without old experiments, outputs, logs, caches, or `.DS_Store`. Copy `pretrained/birna-bert-model` recursively so the new project has no sibling runtime dependency.

- [ ] **Step 3: Verify pretrained portability**

Run:

```bash
shasum -a 256 pretrained/birna-bert-model/pytorch_model.bin
test -f pretrained/birna-bert-model/tokenizer.json
test -f pretrained/birna-bert-model/config.json
```

Expected: destination SHA-256 exactly matches the source and both tests exit 0.

- [ ] **Step 4: Add a source-isolation smoke test**

Create a test that scans project `.py` and configuration files and asserts no runtime path or import contains `../BiRNA_m6A` or the absolute source directory.

- [ ] **Step 5: Commit**

```bash
git add .gitignore README.md requirements.txt train.py configs experiments src pretrained tests
git commit -m "chore: migrate standalone v9a runtime and model assets"
```

### Task 2: Establish `v1_baseline` configuration with exact v9a architecture

**Files:**
- Create: `configs/configarg.py`
- Create: `experiments/v1_baseline/config_v1.py`
- Create: `experiments/v1_baseline/README.md`
- Test: `tests/test_model_parity.py`

- [ ] **Step 1: Write a failing configuration parity test**

The test must compare the new model override dictionary with these v9a invariants:

```python
EXPECTED = {
    "use_center_pooling": False,
    "use_bpe_view": False,
    "use_film": True,
    "film_global_view": "nuc",
    "film_nuc_pooling": "center_cnn_mean",
    "local_window_radius": 3,
    "cnn_kernel_sizes": [3, 5, 7],
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["Wqkv"],
    "freeze_backbone": True,
    "use_handcrafted_features": True,
    "handcrafted_feature_names": ["onehot", "ncp", "eiip", "enac"],
    "handcrafted_cnn_channels": 64,
    "handcrafted_output_dim": 128,
}
```

It must also assert `version_name == "v1_baseline"`, `folds == 5`, and `eval_protocol == "strict_cv"`.

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
pytest tests/test_model_parity.py -v
```

Expected: FAIL because the new config/registry does not exist.

- [ ] **Step 3: Implement the single-version configuration**

Port the v9a overrides exactly, changing only provenance/version and strict protocol fields. Reject unknown versions and reject any `test_as_val` protocol.

- [ ] **Step 4: Add model signature parity**

Instantiate the source and migrated model classes with the same lightweight monkeypatched backbone, then assert identical class hierarchy, named parameter keys, parameter shapes, and forward output shape for identical synthetic inputs.

- [ ] **Step 5: Run parity tests**

Run:

```bash
pytest tests/test_model_parity.py -v
```

Expected: all parity tests PASS.

- [ ] **Step 6: Commit**

```bash
git add configs experiments tests/test_model_parity.py
git commit -m "feat: define v1 baseline with v9a architecture parity"
```

### Task 3: Migrate and rename all MKE-aligned datasets

**Files:**
- Create: `scripts/prepare_data.py`
- Create: `scripts/audit_data.py`
- Create: `data/m6a_41nt/manifest.csv`
- Create: `data/m6a_41nt/*/benchmark.csv`
- Create: `data/m6a_41nt/*/independent_test.csv`
- Test: `tests/test_data_migration.py`

- [ ] **Step 1: Write failing migration parity tests**

For all 11 canonical datasets, assert:

```python
Counter(new_benchmark_rows) == Counter(old_train_rows)
Counter(new_independent_rows) == Counter(old_test_rows)
all(len(sequence) == 41 for sequence, _ in rows)
all(sequence[20] == "A" for sequence, _ in rows)
set(labels) == {0, 1}
```

Also assert `mouse_testis` maps to old `Mouse_test` and that the manifest counts equal the files.

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
pytest tests/test_data_migration.py -v
```

Expected: FAIL because semantic data files do not exist.

- [ ] **Step 3: Implement deterministic migration and manifest generation**

Copy rows without resampling or relabeling. Preserve row order and convert dataset directory names only. Write manifest columns for dataset alias, species, tissue, benchmark/independent paths and counts, sequence length, and validation status.

- [ ] **Step 4: Implement data audit**

Write `data_audit.json` with length, alphabet, center-A, class balance, duplicate-row, and cross-split exact-overlap results. Report inherited overlaps without deleting them.

- [ ] **Step 5: Generate and verify all datasets**

Run:

```bash
python scripts/prepare_data.py --source ../BiRNA_m6A/data/m6A_41bp --destination data/m6a_41nt
python scripts/audit_data.py --data-root data/m6a_41nt
pytest tests/test_data_migration.py -v
```

Expected: 11 datasets generated; no records skipped; all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts data tests/test_data_migration.py
git commit -m "feat: add benchmark and independent test datasets"
```

### Task 4: Extend metrics to match MKE reporting

**Files:**
- Modify: `src/metrics_utils.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write a failing exact-confusion-matrix test**

Use:

```python
labels = [1, 1, 0, 0]
probs = [0.9, 0.4, 0.8, 0.1]
```

Assert `TP=1`, `FN=1`, `FP=1`, `TN=1`, `ACC=0.5`, `Recall=0.5`, `Specificity=0.5`, and finite MCC/AUC/AUPRC.

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
pytest tests/test_metrics.py -v
```

Expected: FAIL because specificity and confusion counts are absent.

- [ ] **Step 3: Implement metrics**

Return `ACC`, `MCC`, `AUC`, `AUPRC`, `F1`, `Precision`, `Recall`, `Specificity`, `TP`, `TN`, `FP`, and `FN`. Continue using probabilities for AUC/AUPRC and threshold 0.5 for class metrics.

- [ ] **Step 4: Verify**

Run:

```bash
pytest tests/test_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/metrics_utils.py tests/test_metrics.py
git commit -m "feat: add MKE-compatible evaluation metrics"
```

### Task 5: Implement benchmark summaries and five-fold independent soft voting

**Files:**
- Create: `src/reporting.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing fold-summary and ensemble tests**

Create five small prediction files containing stable `sample_id`, `sequence`, `label`, and distinct probabilities. Assert:

- benchmark metrics are summarized as mean and sample standard deviation across held-out folds;
- independent probabilities equal the row-wise mean of all five files;
- final classes use threshold 0.5;
- shuffled file order is realigned by `sample_id`, sequence, and label;
- missing, duplicate, mismatched, or fewer-than-five fold rows raise `ValueError`.

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
pytest tests/test_reporting.py -v
```

Expected: FAIL because reporting functions do not exist.

- [ ] **Step 3: Implement reporting**

Provide focused functions to write per-fold benchmark metrics, compute mean/std, align five independent prediction files, average probabilities, calculate ensemble metrics, and save CSV/JSON outputs.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_reporting.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reporting.py tests/test_reporting.py
git commit -m "feat: add five-fold benchmark and soft-vote reporting"
```

### Task 6: Implement paper-ready ROC and precision-recall plots

**Files:**
- Create: `src/plotting.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write failing plotting tests**

Use a balanced synthetic label/probability vector. Assert that plotting creates non-empty PNG, PDF, ROC CSV, and PR CSV files; CSV endpoints and AUC/AP values match scikit-learn; plot labels include AUROC and AUPR.

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
pytest tests/test_plotting.py -v
```

Expected: FAIL because the plotting module does not exist.

- [ ] **Step 3: Implement plotting**

Use a non-interactive Matplotlib backend. Produce a side-by-side ROC/PR figure from ensemble labels and probabilities, save PNG at 300 DPI and vector PDF, and save all curve coordinates and scalar areas.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_plotting.py -v
```

Expected: PASS and all generated artifacts are non-empty.

- [ ] **Step 5: Commit**

```bash
git add src/plotting.py tests/test_plotting.py
git commit -m "feat: export paper-ready ROC and PR figures"
```

### Task 7: Implement strict five-fold orchestration and safe checkpoint deletion

**Files:**
- Modify: `src/train_cv.py`
- Modify: `src/training_utils.py`
- Modify: `train.py`
- Test: `tests/test_cv_protocol.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing protocol tests**

Assert that:

- only `benchmark.csv` is passed to `StratifiedKFold(5, shuffle=True, random_state=seed)`;
- every benchmark sample appears in validation exactly once;
- `independent_test.csv` is never passed to training or selection loaders;
- best epoch is selected by validation ACC;
- the checkpoint remains if prediction export validation fails;
- the checkpoint is deleted only after both prediction files and metrics are successfully re-read;
- fold completion deletes the model object and calls CUDA cache cleanup when CUDA is available.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/test_cv_protocol.py tests/test_reporting.py -v
```

Expected: FAIL against the migrated v9a orchestration.

- [ ] **Step 3: Implement strict orchestration**

Remove `test_as_val` branches. Train folds sequentially, save the best ACC checkpoint, export held-out benchmark and independent predictions, validate exports, delete only the temporary fold checkpoint, release model references, and empty CUDA cache. After fold five, write benchmark mean/std, independent soft-vote metrics/predictions, and ROC/PR outputs.

- [ ] **Step 4: Run protocol tests**

Run:

```bash
pytest tests/test_cv_protocol.py tests/test_reporting.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train.py src/train_cv.py src/training_utils.py tests/test_cv_protocol.py tests/test_reporting.py
git commit -m "feat: add strict five-fold training and checkpoint cleanup"
```

### Task 8: Documentation, dry run, and end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `experiments/v1_baseline/README.md`
- Create: `tests/test_source_isolation.py`

- [ ] **Step 1: Document server setup and exact commands**

Document environment creation, model/data validation, one-dataset training, all-dataset commands, output interpretation, deleted checkpoint behavior, and replotting from saved ensemble predictions.

- [ ] **Step 2: Run static and unit verification**

Run:

```bash
python -m compileall -q configs experiments src scripts train.py
pytest -q
```

Expected: compile exit 0 and all tests PASS.

- [ ] **Step 3: Verify source isolation and weight parity**

Run:

```bash
pytest tests/test_source_isolation.py tests/test_model_parity.py tests/test_data_migration.py -v
shasum -a 256 pretrained/birna-bert-model/pytorch_model.bin
```

Expected: tests PASS and the weight hash matches Task 1.

- [ ] **Step 4: Run a CPU smoke test**

Run a one-epoch, one-fold internal smoke configuration using a temporary synthetic dataset and monkeypatched lightweight backbone. Expected outputs: validated benchmark prediction, independent prediction, metrics, ensemble-ready schema, plot files, and no temporary checkpoint after success.

- [ ] **Step 5: Inspect repository state**

Run:

```bash
git status --short
find . -name best_model.pt -o -name '*.tmp'
```

Expected: only intentional changes are present; no temporary checkpoint or temporary file remains from the smoke test.

- [ ] **Step 6: Commit**

```bash
git add README.md experiments/v1_baseline/README.md tests
git commit -m "docs: add standalone training and reporting guide"
```

