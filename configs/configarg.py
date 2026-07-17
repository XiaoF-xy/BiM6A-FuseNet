from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "m6a_41nt"
MODEL_ROOT = PROJECT_ROOT / "pretrained"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


DATASET_ALIASES = {
    "H_b": "human_brain",
    "H_k": "human_kidney",
    "H_l": "human_liver",
    "M_b": "mouse_brain",
    "M_h": "mouse_heart",
    "M_k": "mouse_kidney",
    "M_l": "mouse_liver",
    "M_t": "mouse_testis",
    "R_b": "rat_brain",
    "R_k": "rat_kidney",
    "R_l": "rat_liver",
}


BASE_VERSION_CONFIG_MODULES = {
    "v1_baseline": "experiments.v1_baseline.config_v1",
    "v1b_proj256_concat": "experiments.v1b_proj256_concat.config_v1b",
}

VERSION_CONFIG_MODULES = dict(BASE_VERSION_CONFIG_MODULES)


def canonical_dataset_name(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset, dataset)


def get_active_data_dir(dataset: str) -> Path:
    return DATA_ROOT / canonical_dataset_name(dataset)


def get_output_dir(version: str, dataset: str, seed: int) -> Path:
    return OUTPUT_ROOT / version / canonical_dataset_name(dataset) / f"seed_{seed}"


@dataclass
class ModelConfig:
    model_dir: Path = MODEL_ROOT / "birna-bert-model"
    tokenizer_dir: Path = MODEL_ROOT / "birna-bert-model"
    freeze_backbone: bool = True
    use_center_pooling: bool = True
    use_bpe_view: bool = False
    use_film: bool = False
    film_global_view: str = "bpe"
    local_window_radius: int = 3
    film_nuc_pooling: str = "center_mean"
    cnn_kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["Wqkv"])
    use_handcrafted_features: bool = False
    handcrafted_feature_names: list[str] = field(default_factory=lambda: ["onehot", "ncp", "eiip", "enac"])
    handcrafted_only: bool = False
    handcrafted_cnn_channels: int = 64
    handcrafted_output_dim: int = 128
    use_gated_fusion: bool = False
    use_projected_concat: bool = False
    gated_fusion_dim: int = 256
    gated_hidden_dim: int = 128


@dataclass
class DataConfig:
    dataset_name: str = "human_brain"
    data_dir: Path = DATA_ROOT / "human_brain"
    sequence_length: int = 41


@dataclass
class TrainConfig:
    output_dir: Path = OUTPUT_ROOT / "v1_baseline" / "Human_Brain" / "seed_42"
    eval_protocol: str = "strict_cv"
    selection_metric: str = "ACC"
    folds: int = 5
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    seed: int = 42
    max_length: int = 64


@dataclass
class ExperimentConfig:
    version_name: str = "v1_baseline"
    description: str = "BiRNA-BERT NUC frozen baseline"


@dataclass
class ProjectConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_overrides(config: ProjectConfig, overrides: dict[str, dict[str, Any]]) -> ProjectConfig:
    for section_name, section_values in overrides.items():
        current_section = getattr(config, section_name)
        setattr(config, section_name, replace(current_section, **section_values))
    return config


def load_experiment_config(version_name: str, dataset_name: str = "Human_Brain", seed: int = 42) -> ProjectConfig:
    if version_name not in VERSION_CONFIG_MODULES:
        supported = ", ".join(sorted(VERSION_CONFIG_MODULES))
        raise ValueError(f"Unknown version: {version_name}. Supported versions: {supported}")
    if seed != 42:
        raise ValueError("v1-family experiments fix the CV split seed at 42 and fold training seeds at 42–46.")

    dataset = canonical_dataset_name(dataset_name)
    module = importlib.import_module(VERSION_CONFIG_MODULES[version_name])
    config = apply_overrides(ProjectConfig(), module.get_overrides(dataset_name=dataset, seed=seed))

    description = config.experiment.description
    config.experiment = replace(config.experiment, version_name=version_name, description=description)
    config.data = replace(
        config.data,
        dataset_name=dataset,
        data_dir=get_active_data_dir(dataset),
    )
    training_updates = {
        "seed": seed,
        "output_dir": get_output_dir(version_name, dataset, seed),
        "eval_protocol": "strict_cv",
    }
    config.training = replace(
        config.training,
        **training_updates,
    )
    return config


def ensure_output_dirs(config: ProjectConfig) -> None:
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)
