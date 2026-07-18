from __future__ import annotations

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
