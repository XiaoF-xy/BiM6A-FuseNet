from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.configarg import (  # noqa: E402
    load_experiment_config,
    ensure_output_dirs,
    VERSION_CONFIG_MODULES,
)


RUNNABLE_VERSIONS = set(VERSION_CONFIG_MODULES)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    return value


def write_resolved_config(config) -> Path:
    config_path = Path(config.training.output_dir) / "resolved_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False, default=_json_default)
    return config_path


def build_cv_command(config) -> list[str]:
    model = config.model
    data = config.data
    training = config.training
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train_cv.py"),
        "--model_dir",
        str(model.model_dir),
        "--tokenizer_dir",
        str(model.tokenizer_dir),
        "--data_dir",
        str(data.data_dir),
        "--output_dir",
        str(training.output_dir),
        "--folds",
        str(training.folds),
        "--epochs",
        str(training.epochs),
        "--batch_size",
        str(training.batch_size),
        "--lr",
        str(training.lr),
        "--weight_decay",
        str(training.weight_decay),
        "--seed",
        str(training.seed),
        "--max_length",
        str(training.max_length),
        "--selection_metric",
        training.selection_metric,
    ]
    if training.optimizer != "adamw":
        command.extend(["--optimizer", training.optimizer])
    if training.scheduler_patience is not None:
        command.extend([
            "--scheduler_patience",
            str(training.scheduler_patience),
            "--scheduler_factor",
            str(training.scheduler_factor),
        ])
    if training.early_stopping_patience is not None:
        command.extend(["--early_stopping_patience", str(training.early_stopping_patience)])
    if model.freeze_backbone:
        command.append("--freeze_backbone")
    if not model.use_center_pooling:
        command.append("--disable_center_pooling")
    if model.use_bpe_view:
        command.append("--use_bpe_view")
    if model.use_film:
        command.extend([
            "--use_film",
            "--film_global_view",
            model.film_global_view,
            "--local_window_radius",
            str(model.local_window_radius),
            "--film_nuc_pooling",
            model.film_nuc_pooling,
            "--cnn_kernel_sizes",
            ",".join(str(kernel) for kernel in model.cnn_kernel_sizes),
        ])
    if model.use_lora:
        command.extend([
            "--use_lora",
            "--lora_r",
            str(model.lora_r),
            "--lora_alpha",
            str(model.lora_alpha),
            "--lora_dropout",
            str(model.lora_dropout),
            "--lora_target_modules",
            ",".join(model.lora_target_modules),
        ])
    if model.use_official_mke_handcrafted:
        command.extend([
            "--use_official_mke_handcrafted",
            "--handcrafted_feature_names",
            ",".join(model.handcrafted_feature_names),
            "--model_label",
            config.experiment.plot_label,
        ])
    elif model.use_handcrafted_features:
        command.extend([
            "--use_handcrafted_features",
            "--handcrafted_feature_names",
            ",".join(model.handcrafted_feature_names),
            "--handcrafted_cnn_channels",
            str(model.handcrafted_cnn_channels),
            "--handcrafted_output_dim",
            str(model.handcrafted_output_dim),
        ])
    if model.use_gated_fusion:
        command.extend([
            "--use_gated_fusion",
            "--gated_fusion_dim",
            str(model.gated_fusion_dim),
            "--gated_hidden_dim",
            str(model.gated_hidden_dim),
        ])
    if model.use_projected_concat:
        command.append("--use_projected_concat")
    if model.use_mke_handcrafted:
        command.extend([
            "--use_mke_handcrafted",
            "--fusion_dim_policy",
            model.fusion_dim_policy,
            "--model_label",
            config.experiment.plot_label,
        ])
        if model.use_full_mke_eca:
            command.append("--use_full_mke_eca")
    if model.handcrafted_only:
        command.append("--handcrafted_only")
    return command


def parse_args(default_version: str = "v1_baseline"):
    parser = argparse.ArgumentParser(description="Run a versioned BiM6A-FuseNet experiment from Python configs.")
    parser.add_argument("--version", type=str, default=default_version)
    parser.add_argument("--dataset", type=str, default="H_b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Print resolved command without launching training.")
    return parser.parse_args()


def main(default_version: str = "v1_baseline"):
    args = parse_args(default_version=default_version)
    config = load_experiment_config(version_name=args.version, dataset_name=args.dataset, seed=args.seed)
    if config.experiment.version_name not in RUNNABLE_VERSIONS:
        raise NotImplementedError(
            f"{config.experiment.version_name} is not runnable. Supported versions: "
            f"{', '.join(sorted(RUNNABLE_VERSIONS))}"
        )

    command = build_cv_command(config)

    print(f"version: {config.experiment.version_name}")
    print(f"dataset: {config.data.dataset_name}")
    print(f"seed: {config.training.seed}")
    print(f"eval_protocol: {config.training.eval_protocol}")
    print(f"selection_metric: {config.training.selection_metric}")
    print(f"output_dir: {config.training.output_dir}")
    print("command:")
    print(" ".join(command))

    if args.dry_run:
        return
    ensure_output_dirs(config)
    resolved_config_path = write_resolved_config(config)
    print(f"resolved_config: {resolved_config_path}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
