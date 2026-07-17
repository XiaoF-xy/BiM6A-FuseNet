"""Safe lifecycle for large temporary fold checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from reporting import read_prediction_file


def delete_checkpoint_after_verified_export(
    checkpoint_path: Path,
    prediction_paths: list[Path],
    metrics_path: Path,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Temporary fold checkpoint is missing: {checkpoint_path}")
    for prediction_path in prediction_paths:
        read_prediction_file(Path(prediction_path))
    with Path(metrics_path).open("r", encoding="utf-8") as handle:
        json.load(handle)
    checkpoint_path.unlink()
    if checkpoint_path.exists():
        raise RuntimeError(f"Temporary fold checkpoint could not be deleted: {checkpoint_path}")
