from __future__ import annotations

from dataclasses import asdict

import pytest

from configs.configarg import load_experiment_config
from scripts.train import build_cv_command


def test_v0a_is_pure_nuc_wqkv_lora():
    config = load_experiment_config("v0a_birna_nuc_lora", "H_b", seed=42)
    command = build_cv_command(config)

    assert config.model.use_birna_single_branch is True
    assert config.model.use_handcrafted_features is False
    assert config.model.use_film is False
    assert config.model.use_bpe_view is False
    assert config.model.use_lora is True
    assert config.model.freeze_backbone is True
    assert config.model.use_center_pooling is False
    assert config.training.epochs == 20
    assert config.training.batch_size == 32
    assert config.training.lr == pytest.approx(1e-4)
    assert config.training.warmup_ratio is None
    assert "--use_birna_single_branch" in command
    assert "--use_lora" in command
    assert "--use_film" not in command
    assert "--use_handcrafted_features" not in command


def test_v0b_is_pure_nuc_full_finetuning_with_warmup():
    config = load_experiment_config("v0b_birna_nuc_fullft", "H_b", seed=42)
    command = build_cv_command(config)

    assert config.model.use_birna_single_branch is True
    assert config.model.use_lora is False
    assert config.model.freeze_backbone is False
    assert config.model.use_center_pooling is False
    assert config.training.epochs == 10
    assert config.training.batch_size == 64
    assert config.training.lr == pytest.approx(1e-6)
    assert config.training.warmup_ratio == pytest.approx(0.1)
    assert "--use_birna_single_branch" in command
    assert command[command.index("--warmup_ratio") + 1] == "0.1"
    assert "--freeze_backbone" not in command
    assert "--use_lora" not in command


def test_v0e_changes_only_v0a_optimizer_to_loraplus_groups():
    baseline = load_experiment_config("v0a_birna_nuc_lora", "H_b", seed=42)
    candidate = load_experiment_config("v0e_birna_nuc_loraplus", "H_b", seed=42)
    baseline_command = build_cv_command(baseline)
    command = build_cv_command(candidate)

    assert candidate.model.lora_target_modules == ["Wqkv"]
    assert candidate.training.use_loraplus is True
    assert candidate.training.lora_a_lr == pytest.approx(5e-5)
    assert candidate.training.lora_b_lr == pytest.approx(8e-4)
    assert candidate.training.classifier_lr == pytest.approx(1e-4)

    assert asdict(candidate.model) == asdict(baseline.model)
    baseline_training = asdict(baseline.training)
    candidate_training = asdict(candidate.training)
    baseline_training.pop("output_dir")
    candidate_training.pop("output_dir")
    for field in ("use_loraplus", "lora_a_lr", "lora_b_lr", "classifier_lr"):
        baseline_training.pop(field)
        candidate_training.pop(field)
    assert candidate_training == baseline_training

    assert "--use_loraplus" in command
    assert "--use_loraplus" not in baseline_command
    assert command[command.index("--lora_a_lr") + 1] == "5e-05"
    assert command[command.index("--lora_b_lr") + 1] == "0.0008"
    assert command[command.index("--classifier_lr") + 1] == "0.0001"


@pytest.mark.parametrize(
    ("version", "expected_targets"),
    [
        ("v0c_birna_lora_full_attention", ["Wqkv", "attention.output.dense"]),
        (
            "v0d_birna_lora_attention_ffn",
            ["Wqkv", "attention.output.dense", "gated_layers", "wo"],
        ),
    ],
)
def test_lora_target_expansion_versions_only_change_target_coverage(
    version,
    expected_targets,
):
    baseline = load_experiment_config("v0a_birna_nuc_lora", "H_b", seed=42)
    candidate = load_experiment_config(version, "H_b", seed=42)
    command = build_cv_command(candidate)

    baseline_model = asdict(baseline.model)
    candidate_model = asdict(candidate.model)
    assert baseline_model.pop("lora_target_modules") == ["Wqkv"]
    assert candidate_model.pop("lora_target_modules") == expected_targets
    assert candidate_model == baseline_model

    baseline_training = asdict(baseline.training)
    candidate_training = asdict(candidate.training)
    baseline_training.pop("output_dir")
    candidate_training.pop("output_dir")
    assert candidate_training == baseline_training
    assert command[command.index("--lora_target_modules") + 1] == ",".join(expected_targets)


@pytest.mark.parametrize(
    "version",
    [
        "v1_baseline",
        "v1b_proj256_concat",
        "v2a_mke_res_eca_native",
        "v2b_mke_res_eca_proj256",
        "v2c_mke_handcrafted_only_official4c",
        "v3a_full_mke_eca_native",
        "v3b_full_mke_eca_proj256",
    ],
)
def test_single_branch_options_do_not_leak_into_existing_versions(version):
    config = load_experiment_config(version, "H_b", seed=42)
    command = build_cv_command(config)

    assert config.model.use_birna_single_branch is False
    assert config.training.warmup_ratio is None
    assert "--use_birna_single_branch" not in command
    assert "--warmup_ratio" not in command
