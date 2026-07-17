from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics_utils import compute_binary_metrics  # noqa: E402
from reporting import ensemble_prediction_files, summarize_metrics  # noqa: E402


def write_predictions(path: Path, probabilities: list[float]) -> None:
    labels = [0, 0, 1, 1]
    sequences = ["C" * 20 + "A" + "C" * 20] * 4
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence", "label", "prob", "pred"])
        writer.writeheader()
        for index, (sequence, label, probability) in enumerate(zip(sequences, labels, probabilities)):
            writer.writerow(
                {
                    "sample_id": index,
                    "sequence": sequence,
                    "label": label,
                    "prob": probability,
                    "pred": int(probability >= 0.5),
                }
            )


def test_binary_metrics_include_paper_metrics_and_confusion_counts():
    metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.7])
    assert metrics["TP"] == 2
    assert metrics["TN"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 0
    assert metrics["Specificity"] == pytest.approx(0.5)
    assert metrics["Sensitivity"] == pytest.approx(1.0)
    assert metrics["ACC"] == pytest.approx(0.75)
    assert set(["MCC", "AUC", "AUPRC", "F1", "Precision", "Recall"]) <= set(metrics)


def test_five_model_soft_voting_averages_probabilities(tmp_path: Path):
    files = []
    fold_probs = [
        [0.1, 0.4, 0.6, 0.9],
        [0.2, 0.3, 0.7, 0.8],
        [0.1, 0.2, 0.8, 0.9],
        [0.2, 0.4, 0.6, 0.8],
        [0.4, 0.2, 0.9, 0.7],
    ]
    for index, probabilities in enumerate(fold_probs, start=1):
        path = tmp_path / f"fold_{index}.csv"
        write_predictions(path, probabilities)
        files.append(path)

    rows, metrics = ensemble_prediction_files(files)

    assert len(rows) == 4
    assert rows[0]["prob"] == pytest.approx(0.2)
    assert rows[2]["prob"] == pytest.approx(0.72)
    assert [row["pred"] for row in rows] == [0, 0, 1, 1]
    assert metrics["ACC"] == pytest.approx(1.0)


def test_soft_voting_requires_exactly_five_aligned_files(tmp_path: Path):
    paths = []
    for index in range(4):
        path = tmp_path / f"fold_{index}.csv"
        write_predictions(path, [0.1, 0.2, 0.8, 0.9])
        paths.append(path)
    with pytest.raises(ValueError, match="exactly five"):
        ensemble_prediction_files(paths)

    fifth = tmp_path / "fold_5.csv"
    write_predictions(fifth, [0.1, 0.2, 0.8, 0.9])
    with fifth.open("r", encoding="utf-8") as handle:
        content = handle.read().replace("0,", "99,", 1)
    fifth.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="aligned"):
        ensemble_prediction_files(paths + [fifth])


def test_cross_validation_summary_uses_sample_standard_deviation():
    summary = summarize_metrics([{"ACC": 0.8}, {"ACC": 1.0}])
    assert summary["mean"]["ACC"] == pytest.approx(0.9)
    assert summary["std"]["ACC"] == pytest.approx(2**0.5 / 10)
