"""Paper-ready ROC/PR plot and coordinate export."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _write_coordinates(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _finish_figure(fig, png_path: Path, pdf_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_paper_curves(predictions: list[dict], output_dir: Path, model_label: str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([int(row["label"]) for row in predictions], dtype=int)
    probabilities = np.asarray([float(row["prob"]) for row in predictions], dtype=float)
    if len(np.unique(labels)) != 2:
        raise ValueError("ROC/PR curves require both negative and positive samples")

    fpr, tpr, roc_thresholds = roc_curve(labels, probabilities)
    precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities)
    auroc = float(roc_auc_score(labels, probabilities))
    auprc = float(average_precision_score(labels, probabilities))

    roc_csv = output_dir / "independent_roc_curve.csv"
    pr_csv = output_dir / "independent_pr_curve.csv"
    _write_coordinates(
        roc_csv,
        ["fpr", "tpr", "threshold"],
        [
            {"fpr": x, "tpr": y, "threshold": threshold}
            for x, y, threshold in zip(fpr, tpr, roc_thresholds)
        ],
    )
    padded_pr_thresholds = list(pr_thresholds) + [float("nan")]
    _write_coordinates(
        pr_csv,
        ["recall", "precision", "threshold"],
        [
            {"recall": x, "precision": y, "threshold": threshold}
            for x, y, threshold in zip(recall, precision, padded_pr_thresholds)
        ],
    )

    combined_png = output_dir / "independent_roc_pr.png"
    combined_pdf = output_dir / "independent_roc_pr.pdf"
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(12.4, 5.2))
    roc_ax.plot(fpr, tpr, color="#d62728", linewidth=2.2, label=f"{model_label} (AUROC={auroc:.4f})")
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.3)
    roc_ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1.01))
    roc_ax.set_title("Independent-test ROC Curve")
    roc_ax.grid(alpha=0.2)
    roc_ax.legend(loc="lower right", frameon=True)

    pr_ax.plot(recall, precision, color="#d62728", linewidth=2.2, label=f"{model_label} (AUPRC={auprc:.4f})")
    pr_ax.axhline(float(labels.mean()), linestyle="--", color="#888888", linewidth=1.3, label="Positive prevalence")
    pr_ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    pr_ax.set_title("Independent-test Precision–Recall Curve")
    pr_ax.grid(alpha=0.2)
    pr_ax.legend(loc="lower left", frameon=True)
    _finish_figure(fig, combined_png, combined_pdf)

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "combined_png": str(combined_png),
        "combined_pdf": str(combined_pdf),
        "roc_csv": str(roc_csv),
        "pr_csv": str(pr_csv),
    }
