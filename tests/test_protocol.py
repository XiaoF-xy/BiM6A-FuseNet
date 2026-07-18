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
    assert "--use_mke_handcrafted" not in command
    assert "--use_full_mke_eca" not in command
    assert "--fusion_dim_policy" not in command
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
    assert "--use_mke_handcrafted" not in command
    assert "--use_full_mke_eca" not in command
    assert "--fusion_dim_policy" not in command
    assert command[command.index("--folds") + 1] == "5"
    assert command[command.index("--epochs") + 1] == "20"
    assert "--selection_metric ACC" in joined


@pytest.mark.parametrize(
    ("version", "paper_label", "dimension_policy", "uses_full_mke_eca"),
    [
        ("v2a_mke_res_eca_native", "BiM6A-FuseNet-v2a", "native", False),
        ("v2b_mke_res_eca_proj256", "BiM6A-FuseNet-v2b", "proj256", False),
        ("v3a_full_mke_eca_native", "BiM6A-FuseNet-v3a", "native", True),
        ("v3b_full_mke_eca_proj256", "BiM6A-FuseNet-v3b", "proj256", True),
    ],
)
def test_mke_versions_have_explicit_architecture_and_dimension_policy(
    version,
    paper_label,
    dimension_policy,
    uses_full_mke_eca,
):
    config = load_experiment_config(version, "H_b", seed=42)
    command = build_cv_command(config)
    joined = " ".join(command)

    assert config.model.use_mke_handcrafted is True
    assert config.model.use_full_mke_eca is uses_full_mke_eca
    assert config.model.fusion_dim_policy == dimension_policy
    assert config.model.handcrafted_feature_names == ["onehot", "ncp", "eiip", "enac"]
    assert "--use_mke_handcrafted" in command
    assert command[command.index("--fusion_dim_policy") + 1] == dimension_policy
    assert config.experiment.plot_label == paper_label
    assert command[command.index("--model_label") + 1] == paper_label
    assert ("--use_full_mke_eca" in command) is uses_full_mke_eca
    assert "--use_projected_concat" not in command
    assert "--use_gated_fusion" not in command
    assert command[command.index("--folds") + 1] == "5"
    assert command[command.index("--epochs") + 1] == "20"
    assert "--selection_metric ACC" in joined
    assert config.training.output_dir == ROOT / "outputs" / version / "human_brain" / "seed_42"


def test_mke_version_configs_differ_only_in_stage_policy_and_metadata():
    v2a = load_experiment_config("v2a_mke_res_eca_native", "H_b", seed=42)
    v2b = load_experiment_config("v2b_mke_res_eca_proj256", "H_b", seed=42)
    v3a = load_experiment_config("v3a_full_mke_eca_native", "H_b", seed=42)
    v3b = load_experiment_config("v3b_full_mke_eca_proj256", "H_b", seed=42)

    assert v2a.training.epochs == v2b.training.epochs == v3a.training.epochs == v3b.training.epochs == 20
    assert v2a.training.lr == v2b.training.lr == v3a.training.lr == v3b.training.lr == 1e-4
    assert v2a.model.lora_target_modules == v2b.model.lora_target_modules == ["Wqkv"]
    assert v3a.model.lora_target_modules == v3b.model.lora_target_modules == ["Wqkv"]
    assert v2a.model.use_full_mke_eca is v2b.model.use_full_mke_eca is False
    assert v3a.model.use_full_mke_eca is v3b.model.use_full_mke_eca is True
