# v4c Threshold-Tuned Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leakage-safe v4c that jointly selects alpha and a decision threshold by benchmark OOF ACC.

**Architecture:** Add a dedicated weighted-threshold fitting rule in `src/late_fusion.py`; reuse the existing five-fold meta cross-fit and independent evaluation pipeline. Register one new fusion method in the wrapper and make its artifacts match v4a/v4b.

**Tech Stack:** Python 3.10, NumPy, scikit-learn metrics, pytest.

---

### Task 1: Prove joint alpha-threshold selection

**Files:**
- Modify: `tests/test_late_fusion.py`
- Test: `tests/test_late_fusion.py::test_weighted_threshold_rule_jointly_selects_alpha_and_acc_optimal_threshold`

- [ ] **Step 1: Write a failing test**

```python
rule = fit_weighted_threshold_rule(rows, alpha_grid=[0.0, 0.5, 1.0], threshold_grid=[0.4, 0.5])
assert rule["alpha_handcrafted"] == pytest.approx(0.5)
assert rule["threshold"] == pytest.approx(0.4)
assert rule["training_metrics"]["ACC"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test and observe the missing-rule failure**

Run: `/opt/anaconda3/bin/python -m pytest tests/test_late_fusion.py::test_weighted_threshold_rule_jointly_selects_alpha_and_acc_optimal_threshold -q`

Expected: FAIL because `fit_weighted_threshold_rule` is not exported.

### Task 2: Add the fusion rule and pipeline dispatch

**Files:**
- Modify: `src/late_fusion.py`
- Modify: `scripts/fuse_predictions.py`
- Modify: `scripts/train.py`
- Test: `tests/test_late_fusion.py`
- Test: `tests/test_protocol.py`

- [ ] **Step 1: Implement the exact alpha/threshold grids and deterministic ACC tie-break**

```python
for alpha in np.linspace(0.0, 1.0, 101):
    probabilities = alpha * features[:, 0] + (1.0 - alpha) * features[:, 1]
    for threshold in np.linspace(0.30, 0.70, 41):
        metrics = compute_binary_metrics(labels, probabilities, threshold=float(threshold))
```

- [ ] **Step 2: Route `weighted_threshold` through meta cross-fit, prediction, metadata, and ACC reporting**

- [ ] **Step 3: Register `v4c_oof_weighted_threshold_tuned` with method `weighted_threshold` and label `BiM6A-FuseNet-v4c`**

- [ ] **Step 4: Run focused tests**

Run: `/opt/anaconda3/bin/python -m pytest tests/test_late_fusion.py tests/test_protocol.py -q`

Expected: PASS.

### Task 3: Document and verify command behavior

**Files:**
- Modify: `README.md`
- Modify: `experiments/v4c_oof_weighted_threshold_tuned/README.md`
- Test: `tests/test_late_fusion.py::test_run_fusion_experiment_writes_paper_ready_artifacts`

- [ ] **Step 1: Document that v4c is no-GPU, uses only base OOF data to select alpha and threshold, and keeps independent labels out of fitting**

- [ ] **Step 2: Verify syntax and no whitespace errors**

Run: `/opt/anaconda3/bin/python -c "import ast, pathlib; [ast.parse(pathlib.Path(name).read_text()) for name in ('src/late_fusion.py', 'scripts/fuse_predictions.py', 'scripts/train.py')]" && git diff --check`

Expected: exit code 0.
