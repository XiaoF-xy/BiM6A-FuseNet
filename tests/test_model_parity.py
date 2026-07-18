from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from configs.configarg import load_experiment_config
from scripts.verify_portable import REQUIRED


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v1_baseline_preserves_v9a_model_configuration():
    config = load_experiment_config("v1_baseline", "H_b", seed=42)

    assert config.experiment.version_name == "v1_baseline"
    assert config.training.eval_protocol == "strict_cv"
    assert config.training.folds == 5
    assert config.training.selection_metric == "ACC"
    assert config.model.use_center_pooling is False
    assert config.model.use_bpe_view is False
    assert config.model.use_film is True
    assert config.model.film_global_view == "nuc"
    assert config.model.film_nuc_pooling == "center_cnn_mean"
    assert config.model.local_window_radius == 3
    assert config.model.cnn_kernel_sizes == [3, 5, 7]
    assert config.model.use_lora is True
    assert config.model.lora_r == 8
    assert config.model.lora_alpha == 32
    assert config.model.lora_dropout == 0.05
    assert config.model.lora_target_modules == ["Wqkv"]
    assert config.model.freeze_backbone is True
    assert config.model.use_handcrafted_features is True
    assert config.model.handcrafted_feature_names == ["onehot", "ncp", "eiip", "enac"]
    assert config.model.handcrafted_cnn_channels == 64
    assert config.model.handcrafted_output_dim == 128


def test_migrated_v9a_architecture_files_keep_recorded_source_hashes():
    expected = {
        "model_birna_film.py": "99e7d41a7a113a9b1394e2d0b5a4034a768eb71ad14067393aa33b8bc59d4836",
        "model_birna_nuc.py": "2c51dd75d270ff2d3bef86d79ec6f97d6fda008c83902d7a0083aa62ee28e5cc",
        "handcrafted_features.py": "d41451babf14a98c1e6cb9d044298ee5fef740d222ae158821928b8a80d7d9ba",
    }

    for filename, expected_hash in expected.items():
        assert sha256(ROOT / "src" / filename) == expected_hash


def test_pretrained_weight_matches_recorded_birna_bert_hash():
    assert sha256(ROOT / "pretrained" / "birna-bert-model" / "pytorch_model.bin") == (
        "4833ca3207d1908a86acffc84d6435379ab65c8da8f1790065c3c683bdacef3b"
    )


def test_portable_check_requires_all_mke_variant_entrypoints():
    required = {path.relative_to(ROOT).as_posix() for path in REQUIRED}
    assert {
        "src/model_birna_mke.py",
        "src/model_mke_handcrafted.py",
        "experiments/mke_variants_common.py",
        "experiments/v2a_mke_res_eca_native/config_v2a.py",
        "experiments/v2b_mke_res_eca_proj256/config_v2b.py",
        "experiments/v3a_full_mke_eca_native/config_v3a.py",
        "experiments/v3b_full_mke_eca_proj256/config_v3b.py",
    } <= required


def test_v1b_changes_only_projected_concat_configuration():
    v1 = load_experiment_config("v1_baseline", "H_b", seed=42)
    v1b = load_experiment_config("v1b_proj256_concat", "H_b", seed=42)

    assert v1.model.use_projected_concat is False
    assert v1b.model.use_projected_concat is True
    assert v1b.training.folds == v1.training.folds == 5
    assert v1b.training.epochs == v1.training.epochs == 20
    assert v1b.training.selection_metric == v1.training.selection_metric == "ACC"
    assert v1b.training.batch_size == v1.training.batch_size
    assert v1b.training.lr == v1.training.lr
    assert v1b.training.weight_decay == v1.training.weight_decay
    v1_model = asdict(v1.model)
    v1b_model = asdict(v1b.model)
    assert v1_model.pop("use_projected_concat") is False
    assert v1b_model.pop("use_projected_concat") is True
    assert v1b_model == v1_model
