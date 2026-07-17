from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from configs.configarg import load_experiment_config
from scripts.train import build_cv_command


ROOT = Path(__file__).resolve().parents[1]


def test_v1_command_is_strict_five_fold_and_has_no_checkpoint_retention_switch():
    config = load_experiment_config("v1_baseline", "H_b", seed=42)
    command = build_cv_command(config)
    joined = " ".join(command)
    assert command[command.index("--folds") + 1] == "5"
    assert "--selection_metric ACC" in joined
    assert "--eval_protocol" not in command
    assert "--keep_best_model" not in command
    assert "--use_projected_concat" not in command
    assert config.data.data_dir / "benchmark.csv" == ROOT / "data/m6a_41nt/human_brain/benchmark.csv"


def test_seed_42_stratified_folds_cover_benchmark_once_without_overlap():
    path = ROOT / "data" / "m6a_41nt" / "human_brain" / "benchmark.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        labels = np.asarray([int(row["label"]) for row in csv.DictReader(handle)])
    indices = np.arange(len(labels))
    seen_validation = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_indices, validation_indices in splitter.split(indices, labels):
        assert not set(train_indices) & set(validation_indices)
        assert Counter(labels[validation_indices])[0] == Counter(labels[validation_indices])[1]
        seen_validation.extend(validation_indices.tolist())
    assert sorted(seen_validation) == indices.tolist()


def test_v1_rejects_a_seed_that_would_change_the_mke_comparison_protocol():
    with pytest.raises(ValueError, match="seed at 42"):
        load_experiment_config("v1_baseline", "H_b", seed=7)


def test_v1b_command_enables_only_projected_concat_fusion():
    config = load_experiment_config("v1b_proj256_concat", "H_b", seed=42)
    command = build_cv_command(config)
    joined = " ".join(command)

    assert "--use_projected_concat" in command
    assert "--use_gated_fusion" not in command
    assert command[command.index("--folds") + 1] == "5"
    assert command[command.index("--epochs") + 1] == "20"
    assert "--selection_metric ACC" in joined
