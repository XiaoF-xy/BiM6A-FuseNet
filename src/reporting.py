"""Result aggregation utilities for strict CV and five-model soft voting."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from metrics_utils import compute_binary_metrics


SUMMARY_METRICS = [
    "ACC",
    "MCC",
    "AUC",
    "AUPRC",
    "F1",
    "Precision",
    "Recall",
    "Sensitivity",
    "Specificity",
]
PREDICTION_FIELDS = ["sample_id", "sequence", "label", "prob", "pred"]


def read_prediction_file(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "sequence": row["sequence"],
                    "label": int(row["label"]),
                    "prob": float(row["prob"]),
                    "pred": int(row["pred"]),
                }
            )
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Prediction file contains duplicate sample_id values: {path}")
    return rows


def write_prediction_file(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def ensemble_prediction_files(paths: list[Path]) -> tuple[list[dict], dict[str, float]]:
    if len(paths) != 5:
        raise ValueError(f"Five-model soft voting requires exactly five prediction files, got {len(paths)}")
    folds = [read_prediction_file(path) for path in paths]
    reference = folds[0]
    reference_keys = [(row["sample_id"], row["sequence"], row["label"]) for row in reference]
    for path, fold in zip(paths[1:], folds[1:]):
        keys = [(row["sample_id"], row["sequence"], row["label"]) for row in fold]
        if keys != reference_keys:
            raise ValueError(f"Five-model prediction files are not aligned: {path}")

    output = []
    for index, reference_row in enumerate(reference):
        probability = float(np.mean([fold[index]["prob"] for fold in folds]))
        output.append(
            {
                "sample_id": reference_row["sample_id"],
                "sequence": reference_row["sequence"],
                "label": reference_row["label"],
                "prob": probability,
                "pred": int(probability >= 0.5),
            }
        )
    metrics = compute_binary_metrics(
        [row["label"] for row in output],
        [row["prob"] for row in output],
    )
    return output, metrics


def summarize_metrics(metrics_per_fold: list[dict]) -> dict[str, dict[str, float]]:
    summary = {"mean": {}, "std": {}}
    for key in SUMMARY_METRICS:
        values = [
            float(metrics[key])
            for metrics in metrics_per_fold
            if key in metrics and metrics[key] is not None and not math.isnan(float(metrics[key]))
        ]
        if not values:
            summary["mean"][key] = math.nan
            summary["std"][key] = math.nan
            continue
        summary["mean"][key] = float(np.mean(values))
        summary["std"][key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return summary
