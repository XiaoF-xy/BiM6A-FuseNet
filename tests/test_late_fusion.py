from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from late_fusion import (
    align_prediction_rows,
    cross_fit_meta_predictions,
    fit_logistic_rule,
    fit_weighted_rule,
    run_fusion_experiment,
)


SEQUENCE = "C" * 20 + "A" + "C" * 20


def prediction_rows(labels, probabilities):
    return [
        {
            "sample_id": str(index),
            "sequence": SEQUENCE,
            "label": int(label),
            "prob": float(probability),
            "pred": int(probability >= 0.5),
        }
        for index, (label, probability) in enumerate(zip(labels, probabilities))
    ]


def write_prediction_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "sequence", "label", "prob", "pred"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def make_base_output_tree(outputs_root: Path, version: str, probabilities: list[float]) -> None:
    run_dir = outputs_root / version / "human_brain" / "seed_42"
    labels = [0, 0, 1, 1]
    rows = prediction_rows(labels, probabilities)
    for fold in range(1, 6):
        write_prediction_csv(run_dir / f"fold_{fold:02d}" / "benchmark_predictions.csv", rows)
    write_prediction_csv(run_dir / "independent_ensemble_predictions.csv", rows)


def test_align_prediction_rows_matches_keys_not_input_order():
    labels = [0, 1]
    handcrafted = prediction_rows(labels, [0.2, 0.8])
    birna = prediction_rows(labels, [0.3, 0.7])[::-1]

    aligned = align_prediction_rows(handcrafted, birna)

    assert [row["sample_id"] for row in aligned] == ["0", "1"]
    assert [row["prob_handcrafted"] for row in aligned] == [0.2, 0.8]
    assert [row["prob_birna"] for row in aligned] == [0.3, 0.7]


def test_align_prediction_rows_rejects_different_labels():
    handcrafted = prediction_rows([0, 1], [0.2, 0.8])
    birna = prediction_rows([1, 1], [0.2, 0.8])

    with pytest.raises(ValueError, match="not aligned"):
        align_prediction_rows(handcrafted, birna)


def test_weighted_rule_selects_acc_maximizing_probability_average():
    rows = align_prediction_rows(
        prediction_rows([0, 1], [0.6, 0.7]),
        prediction_rows([0, 1], [0.2, 0.4]),
    )

    rule = fit_weighted_rule(rows, alpha_grid=[0.0, 0.5, 1.0])

    assert rule["alpha_handcrafted"] == pytest.approx(0.5)
    assert rule["training_metrics"]["ACC"] == pytest.approx(1.0)


def test_logistic_rule_uses_both_base_probabilities():
    rows = align_prediction_rows(
        prediction_rows([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
        prediction_rows([0, 0, 1, 1], [0.2, 0.4, 0.6, 0.8]),
    )

    rule = fit_logistic_rule(rows, seed=42)
    probabilities = rule.predict(rows)

    assert rule.metadata()["feature_order"] == ["prob_handcrafted", "prob_birna"]
    assert len(rule.metadata()["coefficients"]) == 2
    assert probabilities[0] < probabilities[-1]


def test_meta_cross_fit_never_trains_on_held_out_fold():
    folds = []
    for fold in range(1, 6):
        labels = [0, 0, 1, 1]
        handcrafted = prediction_rows(labels, [0.1, 0.3, 0.7, 0.9])
        birna = prediction_rows(labels, [0.2, 0.4, 0.6, 0.8])
        folds.append(align_prediction_rows(handcrafted, birna))

    rows, fold_results = cross_fit_meta_predictions(folds, method="weighted", seed=42)

    assert len(rows) == 20
    assert len({row["sample_id"] for row in rows}) == 20
    assert [result["fold"] for result in fold_results] == [1, 2, 3, 4, 5]
    assert all(result["meta_train_size"] == 16 for result in fold_results)
    assert all(result["meta_validation_size"] == 4 for result in fold_results)


@pytest.mark.parametrize(
    ("method", "version"),
    [
        ("weighted", "v4a_oof_weighted_late_fusion"),
        ("logistic", "v4b_oof_logistic_stacking"),
    ],
)
def test_run_fusion_experiment_writes_paper_ready_artifacts(tmp_path: Path, method: str, version: str):
    outputs_root = tmp_path / "outputs"
    make_base_output_tree(
        outputs_root,
        "v2c_mke_handcrafted_only_official4c",
        [0.1, 0.6, 0.7, 0.9],
    )
    make_base_output_tree(
        outputs_root,
        "v0a_birna_nuc_lora",
        [0.2, 0.3, 0.6, 0.8],
    )

    payload = run_fusion_experiment(
        method=method,
        outputs_root=outputs_root,
        output_version=version,
        dataset="human_brain",
        seed=42,
        model_label=version,
    )

    output_dir = outputs_root / version / "human_brain" / "seed_42"
    expected = [
        "benchmark_meta_oof_predictions.csv",
        "benchmark_cv_metrics.csv",
        "benchmark_cv_summary.json",
        "independent_ensemble_predictions.csv",
        "independent_ensemble_metrics.json",
        "fusion_model.json",
        "resolved_config.json",
        "plots/independent_roc_curve.csv",
        "plots/independent_pr_curve.csv",
        "plots/independent_roc_pr.png",
        "plots/independent_roc_pr.pdf",
    ]
    assert all((output_dir / relative_path).exists() for relative_path in expected)
    assert payload["output_dir"] == str(output_dir)
    with (output_dir / "resolved_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["independent_labels_used_for_fitting"] is False
    assert config["base_versions"] == {
        "handcrafted": "v2c_mke_handcrafted_only_official4c",
        "birna": "v0a_birna_nuc_lora",
    }
