from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from checkpointing import delete_checkpoint_after_verified_export  # noqa: E402
from reporting import write_prediction_file  # noqa: E402


def valid_prediction_rows() -> list[dict]:
    return [
        {
            "sample_id": "0",
            "sequence": "C" * 20 + "A" + "C" * 20,
            "label": 1,
            "prob": 0.8,
            "pred": 1,
        }
    ]


def test_checkpoint_is_deleted_only_after_all_exports_are_readable(tmp_path: Path):
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"temporary checkpoint")
    benchmark = tmp_path / "benchmark.csv"
    independent = tmp_path / "independent.csv"
    write_prediction_file(benchmark, valid_prediction_rows())
    write_prediction_file(independent, valid_prediction_rows())
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"ACC": 1.0}), encoding="utf-8")

    delete_checkpoint_after_verified_export(checkpoint, [benchmark, independent], metrics)

    assert not checkpoint.exists()


def test_checkpoint_is_retained_when_result_validation_fails(tmp_path: Path):
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"temporary checkpoint")
    malformed_predictions = tmp_path / "malformed.csv"
    malformed_predictions.write_text("bad,data\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}", encoding="utf-8")

    with pytest.raises((KeyError, ValueError)):
        delete_checkpoint_after_verified_export(checkpoint, [malformed_predictions], metrics)

    assert checkpoint.exists()
