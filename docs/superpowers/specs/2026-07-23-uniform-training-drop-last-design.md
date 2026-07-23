# Uniform Training Tail-Batch Handling Design

## Goal

Prevent all neural training versions from passing a one-sample final training batch into BatchNorm, while preserving complete validation and independent-test evaluation.

## Cause

`src/train_cv.py` builds every DataLoader with PyTorch's default `drop_last=False`. On M_h fold 1, the 3,521 training samples and batch size 64 produce a final batch of size 1. `OfficialMKEClassifier` uses `BatchNorm1d`, which cannot compute training statistics from this batch.

## Decision

Add a `drop_last: bool = False` argument to the shared `make_loader` function. Pass `drop_last=True` only for the training loader; pass `False` explicitly for validation and independent-test loaders.

## Consequences

- Every existing and future version using `train_cv.py` receives the same safe training behavior.
- Training can omit fewer than `batch_size` samples per epoch; with shuffled training data, the omitted samples are not fixed.
- Validation, OOF predictions, and independent-test metrics use every sample.
- Model architecture, optimizer, batch size, split protocol, checkpoint metric, and reported evaluation data are unchanged.

## Verification

A regression test will assert that a 65-sample training loader with batch size 64 produces one full batch when `drop_last=True`, whereas evaluation loaders preserve both batches when `drop_last=False`.
