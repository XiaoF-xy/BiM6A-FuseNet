"""Leakage-safe probability-level fusion for completed CV experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression

from metrics_utils import compute_binary_metrics, json_safe_metrics
from plotting import save_paper_curves
from reporting import (
    COUNT_METRICS,
    SUMMARY_METRICS,
    read_prediction_file,
    summarize_metrics,
    write_prediction_file,
)


ALIGNED_FIELDS = (
    "sample_id",
    "sequence",
    "label",
    "prob_handcrafted",
    "prob_birna",
)


def _row_key(row: dict) -> tuple[str, str, int]:
    return str(row["sample_id"]), str(row["sequence"]), int(row["label"])


def align_prediction_rows(handcrafted_rows: list[dict], birna_rows: list[dict]) -> list[dict]:
    """Align two prediction tables without trusting their row order."""

    handcrafted_keys = [_row_key(row) for row in handcrafted_rows]
    birna_map = {_row_key(row): row for row in birna_rows}
    if (
        not handcrafted_rows
        or len(handcrafted_keys) != len(set(handcrafted_keys))
        or len(birna_map) != len(birna_rows)
        or set(handcrafted_keys) != set(birna_map)
    ):
        raise ValueError("Base prediction files are not aligned by sample_id, sequence, and label.")

    aligned = []
    for handcrafted_row, key in zip(handcrafted_rows, handcrafted_keys):
        birna_row = birna_map[key]
        aligned.append(
            {
                "sample_id": key[0],
                "sequence": key[1],
                "label": key[2],
                "prob_handcrafted": float(handcrafted_row["prob"]),
                "prob_birna": float(birna_row["prob"]),
            }
        )
    return aligned


def _arrays(rows: Iterable[dict]) -> tuple[np.ndarray, np.ndarray]:
    rows = list(rows)
    if not rows:
        raise ValueError("Fusion requires at least one aligned prediction row.")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    features = np.asarray(
        [
            [float(row["prob_handcrafted"]), float(row["prob_birna"])]
            for row in rows
        ],
        dtype=float,
    )
    if not np.isfinite(features).all() or np.any((features < 0.0) | (features > 1.0)):
        raise ValueError("Fusion probabilities must be finite values in [0, 1].")
    return labels, features


def fit_weighted_rule(
    rows: list[dict],
    alpha_grid: Iterable[float] | None = None,
) -> dict:
    """Select the handcrafted probability weight using ACC only."""

    labels, features = _arrays(rows)
    grid = np.asarray(
        list(alpha_grid) if alpha_grid is not None else np.linspace(0.0, 1.0, 101),
        dtype=float,
    )
    if grid.size == 0 or not np.isfinite(grid).all() or np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError("alpha_grid must contain finite values in [0, 1].")

    candidates = []
    for alpha in grid:
        probabilities = alpha * features[:, 0] + (1.0 - alpha) * features[:, 1]
        metrics = compute_binary_metrics(labels, probabilities)
        candidates.append((float(metrics["ACC"]), -abs(float(alpha) - 0.5), -float(alpha), float(alpha), metrics))
    _, _, _, alpha, metrics = max(candidates, key=lambda item: item[:3])
    return {
        "method": "weighted_probability_average",
        "alpha_handcrafted": alpha,
        "alpha_birna": 1.0 - alpha,
        "selection_metric": "ACC",
        "threshold": 0.5,
        "training_metrics": metrics,
    }


def fit_weighted_threshold_rule(
    rows: list[dict],
    alpha_grid: Iterable[float] | None = None,
    threshold_grid: Iterable[float] | None = None,
) -> dict:
    """Jointly select a probability weight and decision threshold using ACC only."""

    labels, features = _arrays(rows)
    alphas = np.asarray(
        list(alpha_grid) if alpha_grid is not None else np.linspace(0.0, 1.0, 101),
        dtype=float,
    )
    thresholds = np.asarray(
        list(threshold_grid) if threshold_grid is not None else np.linspace(0.30, 0.70, 41),
        dtype=float,
    )
    if alphas.size == 0 or not np.isfinite(alphas).all() or np.any((alphas < 0.0) | (alphas > 1.0)):
        raise ValueError("alpha_grid must contain finite values in [0, 1].")
    if thresholds.size == 0 or not np.isfinite(thresholds).all() or np.any((thresholds < 0.0) | (thresholds > 1.0)):
        raise ValueError("threshold_grid must contain finite values in [0, 1].")

    candidates = []
    for alpha in alphas:
        probabilities = alpha * features[:, 0] + (1.0 - alpha) * features[:, 1]
        for threshold in thresholds:
            candidates.append(
                (
                    float(np.mean((probabilities >= threshold) == labels)),
                    -abs(float(threshold) - 0.5),
                    -abs(float(alpha) - 0.5),
                    -float(alpha),
                    -float(threshold),
                    float(alpha),
                    float(threshold),
                )
            )
    _, _, _, _, _, alpha, threshold = max(candidates, key=lambda item: item[:5])
    probabilities = alpha * features[:, 0] + (1.0 - alpha) * features[:, 1]
    metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)
    return {
        "method": "weighted_probability_average_threshold_tuned",
        "alpha_handcrafted": alpha,
        "alpha_birna": 1.0 - alpha,
        "selection_metric": "ACC",
        "threshold": threshold,
        "training_metrics": metrics,
    }


@dataclass
class LogisticFusionRule:
    model: LogisticRegression

    def predict(self, rows: list[dict]) -> np.ndarray:
        _, features = _arrays(rows)
        return self.model.predict_proba(features)[:, 1]

    def metadata(self) -> dict:
        return {
            "method": "logistic_stacking",
            "feature_order": ["prob_handcrafted", "prob_birna"],
            "coefficients": [float(value) for value in self.model.coef_[0]],
            "intercept": float(self.model.intercept_[0]),
            "regularization": "l2",
            "C": float(self.model.C),
            "solver": str(self.model.solver),
            "threshold": 0.5,
        }


def fit_logistic_rule(rows: list[dict], seed: int) -> LogisticFusionRule:
    labels, features = _arrays(rows)
    if len(np.unique(labels)) != 2:
        raise ValueError("Logistic stacking requires both binary classes in meta-training data.")
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(features, labels)
    return LogisticFusionRule(model=model)


def _fit_rule(rows: list[dict], method: str, seed: int):
    if method == "weighted":
        return fit_weighted_rule(rows)
    if method == "weighted_threshold":
        return fit_weighted_threshold_rule(rows)
    if method == "logistic":
        return fit_logistic_rule(rows, seed=seed)
    raise ValueError(f"Unknown fusion method: {method}")


def predict_with_rule(rule, rows: list[dict]) -> np.ndarray:
    if isinstance(rule, LogisticFusionRule):
        return rule.predict(rows)
    if isinstance(rule, dict) and rule.get("method") in {
        "weighted_probability_average",
        "weighted_probability_average_threshold_tuned",
    }:
        _, features = _arrays(rows)
        alpha = float(rule["alpha_handcrafted"])
        return alpha * features[:, 0] + (1.0 - alpha) * features[:, 1]
    raise TypeError(f"Unsupported fusion rule: {type(rule).__name__}")


def rule_metadata(rule) -> dict:
    if isinstance(rule, LogisticFusionRule):
        return rule.metadata()
    return dict(rule)


def cross_fit_meta_predictions(
    folds: list[list[dict]],
    method: str,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Create unbiased meta-level OOF predictions by holding out each base OOF fold."""

    if len(folds) != 5 or any(not fold for fold in folds):
        raise ValueError("Meta cross-fitting requires exactly five non-empty benchmark OOF folds.")

    output_rows = []
    fold_results = []
    for held_out_index, validation_rows in enumerate(folds):
        train_rows = [row for index, fold in enumerate(folds) if index != held_out_index for row in fold]
        rule = _fit_rule(train_rows, method=method, seed=seed + held_out_index)
        probabilities = predict_with_rule(rule, validation_rows)
        threshold = float(rule_metadata(rule)["threshold"])
        fused_rows = [
            {
                "sample_id": f"fold_{held_out_index + 1:02d}:{row['sample_id']}",
                "sequence": row["sequence"],
                "label": int(row["label"]),
                "prob": float(probability),
                "pred": int(probability >= threshold),
                "threshold": threshold,
            }
            for row, probability in zip(validation_rows, probabilities)
        ]
        metrics = compute_binary_metrics(
            [row["label"] for row in fused_rows],
            [row["prob"] for row in fused_rows],
            threshold=threshold,
        )
        output_rows.extend(fused_rows)
        fold_results.append(
            {
                "fold": held_out_index + 1,
                "meta_train_size": len(train_rows),
                "meta_validation_size": len(validation_rows),
                "metrics": metrics,
                "rule": rule_metadata(rule),
            }
        )
    return output_rows, fold_results


def _write_meta_cv_csv(path: Path, fold_results: list[dict], summary: dict) -> None:
    fieldnames = ["fold", "meta_train_size", "meta_validation_size"] + SUMMARY_METRICS + COUNT_METRICS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in fold_results:
            writer.writerow(
                {
                    "fold": result["fold"],
                    "meta_train_size": result["meta_train_size"],
                    "meta_validation_size": result["meta_validation_size"],
                    **{
                        key: result["metrics"].get(key, "")
                        for key in SUMMARY_METRICS + COUNT_METRICS
                    },
                }
            )
        writer.writerow({"fold": "mean", **summary["mean"]})
        writer.writerow({"fold": "std", **summary["std"]})


def _load_base_predictions(
    outputs_root: Path,
    version: str,
    dataset: str,
    seed: int,
) -> tuple[list[list[dict]], list[dict], Path]:
    run_dir = outputs_root / version / dataset / f"seed_{seed}"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Required base output directory not found: {run_dir}")
    folds = [
        read_prediction_file(run_dir / f"fold_{fold:02d}" / "benchmark_predictions.csv")
        for fold in range(1, 6)
    ]
    independent = read_prediction_file(run_dir / "independent_ensemble_predictions.csv")
    return folds, independent, run_dir


def run_fusion_experiment(
    *,
    method: str,
    outputs_root: Path,
    output_version: str,
    dataset: str,
    seed: int,
    model_label: str,
    handcrafted_version: str = "v2c_mke_handcrafted_only_official4c",
    birna_version: str = "v0a_birna_nuc_lora",
) -> dict:
    """Fuse two completed strict-CV experiments without retraining base models."""

    outputs_root = Path(outputs_root).expanduser().resolve()
    handcrafted_folds, handcrafted_independent, handcrafted_dir = _load_base_predictions(
        outputs_root, handcrafted_version, dataset, seed
    )
    birna_folds, birna_independent, birna_dir = _load_base_predictions(
        outputs_root, birna_version, dataset, seed
    )
    aligned_folds = [
        align_prediction_rows(handcrafted_fold, birna_fold)
        for handcrafted_fold, birna_fold in zip(handcrafted_folds, birna_folds)
    ]
    aligned_independent = align_prediction_rows(handcrafted_independent, birna_independent)

    output_dir = outputs_root / output_version / dataset / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_versions = {"handcrafted": handcrafted_version, "birna": birna_version}
    base_directories = {"handcrafted": str(handcrafted_dir), "birna": str(birna_dir)}

    meta_oof_rows, fold_results = cross_fit_meta_predictions(
        aligned_folds,
        method=method,
        seed=seed,
    )
    meta_oof_metrics = compute_binary_metrics(
        [row["label"] for row in meta_oof_rows],
        [row["prob"] for row in meta_oof_rows],
        threshold=[row["threshold"] for row in meta_oof_rows],
    )
    fold_summary = summarize_metrics([result["metrics"] for result in fold_results])
    meta_oof_path = output_dir / "benchmark_meta_oof_predictions.csv"
    write_prediction_file(
        meta_oof_path,
        [
            {
                "sample_id": row["sample_id"],
                "sequence": row["sequence"],
                "label": row["label"],
                "prob": row["prob"],
                "pred": row["pred"],
            }
            for row in meta_oof_rows
        ],
    )
    _write_meta_cv_csv(output_dir / "benchmark_cv_metrics.csv", fold_results, fold_summary)
    benchmark_payload = {
        "method": method,
        "eval_protocol": "fold_preserving_meta_cross_fit_on_base_oof",
        "base_versions": base_versions,
        "independent_labels_used_for_fitting": False,
        "folds": fold_results,
        "benchmark_cv_mean": fold_summary["mean"],
        "benchmark_cv_std": fold_summary["std"],
        "benchmark_meta_oof_metrics": meta_oof_metrics,
        "benchmark_meta_oof_predictions": str(meta_oof_path),
    }
    with (output_dir / "benchmark_cv_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(benchmark_payload), handle, indent=2, ensure_ascii=False)

    all_oof_rows = [row for fold in aligned_folds for row in fold]
    final_rule = _fit_rule(all_oof_rows, method=method, seed=seed)
    independent_probabilities = predict_with_rule(final_rule, aligned_independent)
    final_threshold = float(rule_metadata(final_rule)["threshold"])
    independent_rows = [
        {
            "sample_id": row["sample_id"],
            "sequence": row["sequence"],
            "label": int(row["label"]),
            "prob": float(probability),
            "pred": int(probability >= final_threshold),
        }
        for row, probability in zip(aligned_independent, independent_probabilities)
    ]
    independent_metrics = compute_binary_metrics(
        [row["label"] for row in independent_rows],
        [row["prob"] for row in independent_rows],
        threshold=final_threshold,
    )
    independent_path = output_dir / "independent_ensemble_predictions.csv"
    write_prediction_file(independent_path, independent_rows)
    plot_artifacts = save_paper_curves(independent_rows, output_dir / "plots", model_label=model_label)

    final_rule_payload = {
        **rule_metadata(final_rule),
        "trained_on": "all benchmark OOF predictions",
        "meta_training_size": len(all_oof_rows),
        "base_versions": base_versions,
        "base_directories": base_directories,
        "independent_labels_used_for_fitting": False,
    }
    with (output_dir / "fusion_model.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(final_rule_payload), handle, indent=2, ensure_ascii=False)

    independent_payload = {
        "method": method,
        "probability_rule": rule_metadata(final_rule),
        "threshold": final_threshold,
        "base_versions": base_versions,
        "test_set_role": "final_evaluation_only",
        "independent_labels_used_for_fitting": False,
        "metrics": independent_metrics,
        "predictions": str(independent_path),
        "plots": plot_artifacts,
    }
    with (output_dir / "independent_ensemble_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(independent_payload), handle, indent=2, ensure_ascii=False)

    resolved_config = {
        "version": output_version,
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "output_dir": str(output_dir),
        "base_versions": base_versions,
        "base_directories": base_directories,
        "benchmark_protocol": "five-fold base OOF plus fold-preserving meta cross-fit",
        "independent_protocol": "fit final rule on all benchmark OOF, evaluate independent ensembles once",
        "selection_metric": "ACC" if method in {"weighted", "weighted_threshold"} else None,
        "threshold": final_threshold,
        "independent_labels_used_for_fitting": False,
    }
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, indent=2, ensure_ascii=False)

    return {
        "output_dir": str(output_dir),
        "benchmark_meta_oof_metrics": meta_oof_metrics,
        "independent_metrics": independent_metrics,
        "fusion_model": final_rule_payload,
    }
