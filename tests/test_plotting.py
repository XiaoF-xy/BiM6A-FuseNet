from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotting import save_paper_curves  # noqa: E402


def test_paper_curves_export_images_and_reusable_coordinates(tmp_path: Path):
    rows = [
        {"label": 0, "prob": 0.05},
        {"label": 0, "prob": 0.25},
        {"label": 1, "prob": 0.75},
        {"label": 1, "prob": 0.95},
    ]
    artifacts = save_paper_curves(rows, tmp_path, model_label="BiM6A-FuseNet v1")

    for name in ["combined_png", "combined_pdf", "roc_csv", "pr_csv"]:
        path = Path(artifacts[name])
        assert path.exists() and path.stat().st_size > 0

    with Path(artifacts["roc_csv"]).open(newline="", encoding="utf-8") as handle:
        assert set(next(csv.DictReader(handle))) == {"fpr", "tpr", "threshold"}
    with Path(artifacts["pr_csv"]).open(newline="", encoding="utf-8") as handle:
        assert set(next(csv.DictReader(handle))) == {"recall", "precision", "threshold"}
    assert artifacts["AUROC"] == 1.0
    assert artifacts["AUPRC"] == 1.0
