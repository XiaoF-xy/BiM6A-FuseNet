# Uniform Training Tail-Batch Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent one-sample tail batches from reaching BatchNorm during training in every version that uses the shared cross-validation trainer.

**Architecture:** Extend the single shared `make_loader` boundary in `src/train_cv.py` with a `drop_last` option. Training passes `True`; validation and independent testing pass `False`. The change is covered at the DataLoader level, so it applies to every existing experiment version without version-specific conditionals.

**Tech Stack:** Python 3.10, PyTorch DataLoader, pytest.

---

### Task 1: Specify and prove the desired loader behavior

**Files:**
- Modify: `tests/test_mke_official_handcrafted.py`
- Test: `tests/test_mke_official_handcrafted.py::test_shared_training_loader_drops_only_incomplete_tail_batches`

- [ ] **Step 1: Write the failing test**

```python
def test_shared_training_loader_drops_only_incomplete_tail_batches():
    samples = [SequenceSample(sequence="A" * 41, label=index % 2) for index in range(65)]
    train_loader = make_loader(
    samples, tokenizer=None, max_length=64, batch_size=64, shuffle=False,
    use_bpe_view=False, use_official_mke_handcrafted=True, drop_last=True,
    )
    eval_loader = make_loader(
    samples, tokenizer=None, max_length=64, batch_size=64, shuffle=False,
    use_bpe_view=False, use_official_mke_handcrafted=True, drop_last=False,
    )
    assert [len(batch["labels"]) for batch in train_loader] == [64]
    assert [len(batch["labels"]) for batch in eval_loader] == [64, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mke_official_handcrafted.py::test_shared_training_loader_drops_only_incomplete_tail_batches -q`

Expected: FAIL because `make_loader` does not accept `drop_last`.

### Task 2: Implement the common loader option

**Files:**
- Modify: `src/train_cv.py:346-388,511-548`
- Test: `tests/test_mke_official_handcrafted.py::test_shared_training_loader_drops_only_incomplete_tail_batches`

- [ ] **Step 1: Add the parameter and pass it to DataLoader**

```python
def make_loader(..., use_official_mke_handcrafted: bool = False, drop_last: bool = False):
    ...
    return DataLoader(..., drop_last=drop_last)
```

- [ ] **Step 2: Set loader roles explicitly**

```python
# train_loader
drop_last=True,
# val_loader and test_loader
drop_last=False,
```

- [ ] **Step 3: Run the regression test**

Run: `python -m pytest tests/test_mke_official_handcrafted.py::test_shared_training_loader_drops_only_incomplete_tail_batches -q`

Expected: PASS.

### Task 3: Verify shared behavior and documentation

**Files:**
- Modify: `README.md`
- Test: `tests/test_mke_official_handcrafted.py`

- [ ] **Step 1: Document the shared training-only tail-batch policy**

Add a concise note stating that incomplete training batches are dropped, while validation and independent-test sets are evaluated in full.

- [ ] **Step 2: Run relevant tests**

Run: `python -m pytest tests/test_mke_official_handcrafted.py -q`

Expected: PASS.
