from __future__ import annotations

import argparse
import gc
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from dataset_utils import SequenceSample, read_samples_from_file
from data_audit import audit_dataset
from checkpointing import delete_checkpoint_after_verified_export
from metrics_utils import format_metrics, json_safe_metrics
from plotting import save_paper_curves
from reporting import (
    ensemble_prediction_files,
    summarize_metrics,
    write_benchmark_cv_csv,
    write_prediction_file,
)
from model_birna_dual_view import BiRNADualViewClassifier
from handcrafted_features import handcrafted_channel_count, parse_feature_names
from model_birna_film import (
    BiRNAFiLMGatedHandcraftedClassifier,
    BiRNAFiLMHandcraftedClassifier,
    BiRNAFiLMLocalClassifier,
    HandcraftedOnlyClassifier,
)
from model_birna_film_proj import BiRNAFiLMProjectedConcatClassifier
from model_birna_mke import BiRNAFiLMMKEClassifier
from model_birna_nuc import BiRNANucClassifier, load_birna_tokenizer
from training_utils import (
    DualViewDataCollator,
    RNANucDataset,
    NucDataCollator,
    NucViewDataCollator,
    append_train_log,
    evaluate,
    metric_score,
    resolve_path,
    save_predictions,
    select_device,
    set_seed,
    train_one_epoch,
    write_train_log_header,
)


METRIC_KEYS = [
    "ACC", "MCC", "AUC", "AUPRC", "F1", "Precision", "Recall", "Sensitivity", "Specificity"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stratified k-fold cross validation on one 41bp m6A dataset with BiRNA-BERT."
    )
    parser.add_argument("--model_dir", type=Path, default=Path("./pretrained/birna-bert-model"))
    parser.add_argument("--tokenizer_dir", type=Path, default=Path("./pretrained/birna-bert-model"))
    parser.add_argument("--data_dir", type=Path, default=Path("./data/m6a_41nt/human_brain"))
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs/v1_baseline/human_brain/seed_42"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay. Default is 0.01, matching PyTorch AdamW's implicit default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=METRIC_KEYS,
        default="ACC",
        help="Metric used to select the best epoch/checkpoint. Default follows DFM-style ACC selection.",
    )
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument(
        "--disable_center_pooling",
        action="store_true",
        help="Ablation switch: use sequence mean pooling only and remove explicit center-token pooling.",
    )
    parser.add_argument("--use_bpe_view", action="store_true")
    parser.add_argument("--use_film", action="store_true")
    parser.add_argument(
        "--film_global_view",
        type=str,
        choices=["bpe", "nuc"],
        default="bpe",
        help="Global view used to generate FiLM gamma/beta when --use_film is set.",
    )
    parser.add_argument(
        "--local_window_radius",
        type=int,
        default=3,
        help="Radius around the center A for NUC local pooling. radius=3 uses positions 17:24 for 41bp input.",
    )
    parser.add_argument(
        "--film_nuc_pooling",
        type=str,
        choices=["center_mean", "full_mean", "center_cnn_mean", "full_cnn_mean", "full_mean_center_cnn_mean"],
        default="center_mean",
        help=(
            "NUC branch pooled after FiLM modulation. center_mean keeps v6 behavior; "
            "full_mean removes the center window; *_cnn_mean adds a learnable multi-scale CNN before mean pooling; "
            "full_mean_center_cnn_mean fuses full 41bp mean and center-window CNN branches."
        ),
    )
    parser.add_argument(
        "--cnn_kernel_sizes",
        type=str,
        default="3,5,7",
        help="Comma-separated positive odd Conv1d kernel sizes for CNN FiLM NUC branch.",
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="Wqkv")
    parser.add_argument("--use_handcrafted_features", action="store_true")
    parser.add_argument(
        "--handcrafted_feature_names",
        type=str,
        default="onehot,ncp,eiip,enac",
        help="Comma-separated handcrafted feature groups: onehot,ncp,eiip,enac.",
    )
    parser.add_argument(
        "--handcrafted_only",
        action="store_true",
        help="Train only the handcrafted multi-scale CNN branch without BiRNA-BERT.",
    )
    parser.add_argument("--handcrafted_cnn_channels", type=int, default=64)
    parser.add_argument("--handcrafted_output_dim", type=int, default=128)
    parser.add_argument(
        "--use_gated_fusion",
        action="store_true",
        help="Fuse BiRNA and handcrafted branches with a learnable vector gate instead of simple concatenation.",
    )
    parser.add_argument(
        "--use_projected_concat",
        action="store_true",
        help="Project both branches independently to 256 dimensions before concatenation.",
    )
    parser.add_argument(
        "--use_mke_handcrafted",
        action="store_true",
        help="Use the four-stream MKE ResNet-ECA handcrafted encoder.",
    )
    parser.add_argument(
        "--use_full_mke_eca",
        action="store_true",
        help="Add full post-fusion channel and multi-scale spatial MKE-ECA attention.",
    )
    parser.add_argument(
        "--fusion_dim_policy",
        choices=["native", "proj256"],
        default="native",
        help="MKE fusion dimensions: asymmetric native widths or aligned 256-dimensional projections.",
    )
    parser.add_argument(
        "--model_label",
        type=str,
        default="BiM6A-FuseNet-v1",
        help="Model name written into ROC/PR plot legends and result metadata.",
    )
    parser.add_argument("--gated_fusion_dim", type=int, default=256)
    parser.add_argument("--gated_hidden_dim", type=int, default=128)
    return parser.parse_args()


def parse_cnn_kernel_sizes(value: str) -> list[int]:
    try:
        kernels = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"--cnn_kernel_sizes must be comma-separated integers, got: {value}") from exc
    if not kernels:
        raise ValueError("--cnn_kernel_sizes must contain at least one integer.")
    invalid = [kernel for kernel in kernels if kernel <= 0 or kernel % 2 == 0]
    if invalid:
        raise ValueError(
            "--cnn_kernel_sizes must contain only positive odd integers. "
            f"Invalid values: {invalid}; full input: {value}"
        )
    return kernels


def load_single_dataset_train_test(data_dir: Path) -> tuple[list[SequenceSample], list[SequenceSample], dict]:
    train_path = data_dir / "benchmark.csv"
    test_path = data_dir / "independent_test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Dataset benchmark.csv not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Dataset independent_test.csv not found: {test_path}")

    train_samples, train_stats = read_samples_from_file(train_path, expected_length=41)
    test_samples, test_stats = read_samples_from_file(test_path, expected_length=41)
    if not train_samples:
        raise ValueError(f"No valid training samples loaded from: {train_path}")
    if not test_samples:
        raise ValueError(f"No valid independent test samples loaded from: {test_path}")

    stats = {
        "data_dir": str(data_dir),
        "cv_source": str(train_path),
        "independent_test_source": str(test_path),
        "train_raw_records": train_stats["raw_records"],
        "test_raw_records": test_stats["raw_records"],
        "train_skipped": train_stats["skipped"],
        "test_skipped": test_stats["skipped"],
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        "train_label_counts": dict(Counter(sample.label for sample in train_samples)),
        "test_label_counts": dict(Counter(sample.label for sample in test_samples)),
    }
    return train_samples, test_samples, stats


def make_loader(
    samples: list[SequenceSample],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    use_bpe_view: bool,
    use_film: bool = False,
    use_handcrafted_features: bool = False,
    handcrafted_feature_names: list[str] | None = None,
):
    if use_bpe_view:
        collator_cls = DualViewDataCollator
    elif use_film:
        collator_cls = NucViewDataCollator
    else:
        collator_cls = NucDataCollator
    return DataLoader(
        RNANucDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator_cls(
            tokenizer=tokenizer,
            max_length=max_length,
            include_handcrafted=use_handcrafted_features,
            handcrafted_feature_names=handcrafted_feature_names,
        ),
        num_workers=0,
    )


def train_one_fold(
    fold_idx: int,
    train_samples: list[SequenceSample],
    val_samples: list[SequenceSample],
    independent_test_samples: list[SequenceSample],
    tokenizer,
    args,
    device: torch.device,
) -> dict:
    fold_dir = args.output_dir / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = fold_dir / "train_log.csv"
    best_model_path = fold_dir / "best_model.pt"
    write_train_log_header(train_log_path)

    set_seed(args.seed + fold_idx - 1)
    lora_target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    common_model_kwargs = {
        "model_dir": args.model_dir,
        "freeze_backbone": args.freeze_backbone,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": lora_target_modules,
    }
    if args.use_handcrafted_features and not args.use_film and not args.handcrafted_only:
        raise ValueError("--use_handcrafted_features requires --use_film unless --handcrafted_only is set.")
    handcrafted_input_channels = handcrafted_channel_count(args.handcrafted_feature_names)
    if args.handcrafted_only:
        model = HandcraftedOnlyClassifier(
            handcrafted_input_channels=handcrafted_input_channels,
            handcrafted_cnn_channels=args.handcrafted_cnn_channels,
            handcrafted_output_dim=args.handcrafted_output_dim,
            cnn_kernel_sizes=args.cnn_kernel_sizes,
        )
    elif args.use_film:
        if args.use_mke_handcrafted:
            film_model_cls = BiRNAFiLMMKEClassifier
        elif args.use_projected_concat:
            film_model_cls = BiRNAFiLMProjectedConcatClassifier
        elif args.use_gated_fusion:
            film_model_cls = BiRNAFiLMGatedHandcraftedClassifier
        else:
            film_model_cls = BiRNAFiLMHandcraftedClassifier if args.use_handcrafted_features else BiRNAFiLMLocalClassifier
        model_extra_kwargs = {}
        if args.use_mke_handcrafted:
            model_extra_kwargs.update(
                {
                    "handcrafted_feature_names": args.handcrafted_feature_names,
                    "use_full_mke_eca": args.use_full_mke_eca,
                    "fusion_dim_policy": args.fusion_dim_policy,
                }
            )
        elif args.use_handcrafted_features:
            model_extra_kwargs.update(
                {
                    "handcrafted_input_channels": handcrafted_input_channels,
                    "handcrafted_cnn_channels": args.handcrafted_cnn_channels,
                    "handcrafted_output_dim": args.handcrafted_output_dim,
                }
            )
        if args.use_gated_fusion:
            model_extra_kwargs.update(
                {
                    "gated_fusion_dim": args.gated_fusion_dim,
                    "gated_hidden_dim": args.gated_hidden_dim,
                }
            )
        model = film_model_cls(
            **common_model_kwargs,
            film_global_view=args.film_global_view,
            local_window_radius=args.local_window_radius,
            film_nuc_pooling=args.film_nuc_pooling,
            cnn_kernel_sizes=args.cnn_kernel_sizes,
            **model_extra_kwargs,
        )
    else:
        model_cls = BiRNADualViewClassifier if args.use_bpe_view else BiRNANucClassifier
        model = model_cls(
            **common_model_kwargs,
            use_center_pooling=not args.disable_center_pooling,
        )
    model.to(device)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Fold {fold_idx} trainable_params: {trainable_params:,} / total_params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(
        train_samples,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=True,
        use_bpe_view=args.use_bpe_view,
        use_film=args.use_film,
        use_handcrafted_features=args.use_handcrafted_features,
        handcrafted_feature_names=args.handcrafted_feature_names,
    )
    val_loader = make_loader(
        val_samples,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=False,
        use_bpe_view=args.use_bpe_view,
        use_film=args.use_film,
        use_handcrafted_features=args.use_handcrafted_features,
        handcrafted_feature_names=args.handcrafted_feature_names,
    )
    test_loader = make_loader(
        independent_test_samples,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=False,
        use_bpe_view=args.use_bpe_view,
        use_film=args.use_film,
        use_handcrafted_features=args.use_handcrafted_features,
        handcrafted_feature_names=args.handcrafted_feature_names,
    )

    best_score = -math.inf
    best_epoch = None
    selection_desc = "benchmark validation"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            freeze_backbone=args.freeze_backbone,
        )
        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc=f"Fold {fold_idx} epoch {epoch} {selection_desc}",
        )
        score = metric_score(val_metrics, selection_metric=args.selection_metric)
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "best_score": best_score,
                    "selection_metric": args.selection_metric,
                    "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "eval_protocol": "strict_cv",
                    "fold_sizes": {
                        "train": len(train_samples),
                        "val": len(val_samples),
                        "independent_test": len(independent_test_samples),
                    },
                },
                best_model_path,
            )

        print(
            f"Fold {fold_idx} epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"{format_metrics(val_metrics)} best_{args.selection_metric}={best_score:.4f}"
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_score": best_score,
            "is_best": int(is_best),
        }
        row.update(val_metrics)
        append_train_log(train_log_path, row)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, selected_val_metrics, val_predictions = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        desc=f"Fold {fold_idx} selected benchmark validation",
        return_predictions=True,
    )
    benchmark_predictions_path = fold_dir / "benchmark_predictions.csv"
    save_predictions(benchmark_predictions_path, val_predictions)
    test_loss, test_metrics, test_predictions = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        desc=f"Fold {fold_idx} independent test",
        return_predictions=True,
    )
    independent_predictions_path = fold_dir / "independent_predictions.csv"
    save_predictions(independent_predictions_path, test_predictions)

    fold_payload = {
        "fold": fold_idx,
        "eval_protocol": "strict_cv",
        "selection_set": "benchmark.csv held-out fold",
        "test_set_role": "final_evaluation_only",
        "best_epoch": best_epoch,
        "best_score": best_score,
        "selection_metric": args.selection_metric,
        "best_model_path": str(best_model_path),
        "best_model_deleted": False,
        "benchmark_validation_loss": val_loss,
        "benchmark_validation_metrics": selected_val_metrics,
        "independent_test_loss": test_loss,
        "independent_test_metrics": test_metrics,
        "fold_sizes": {
            "train": len(train_samples),
            "val": len(val_samples),
            "independent_test": len(independent_test_samples),
        },
        "label_counts": {
            "train": dict(Counter(sample.label for sample in train_samples)),
            "val": dict(Counter(sample.label for sample in val_samples)),
            "independent_test": dict(Counter(sample.label for sample in independent_test_samples)),
        },
    }
    metrics_path = fold_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(fold_payload), handle, indent=2, ensure_ascii=False)

    # Deletion is deliberately last: helper validates all irreplaceable result files first.
    delete_checkpoint_after_verified_export(
        best_model_path,
        [
            (benchmark_predictions_path, len(val_samples)),
            (independent_predictions_path, len(independent_test_samples)),
        ],
        metrics_path,
    )
    fold_payload["best_model_deleted"] = True
    fold_payload["best_model_path"] = None
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(fold_payload), handle, indent=2, ensure_ascii=False)

    del checkpoint, optimizer, criterion, train_loader, val_loader, test_loader, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"Fold {fold_idx} test_loss={test_loss:.4f} {format_metrics(test_metrics)}")
    print(f"Fold {fold_idx} deleted temporary checkpoint after verified result export: {best_model_path}")
    return fold_payload


def summarize_folds(fold_results: list[dict]) -> dict:
    benchmark = summarize_metrics([result["benchmark_validation_metrics"] for result in fold_results])
    return {"folds": fold_results, "benchmark_cv_mean": benchmark["mean"], "benchmark_cv_std": benchmark["std"]}


def main():
    args = parse_args()
    args.model_dir = resolve_path(args.model_dir)
    args.tokenizer_dir = resolve_path(args.tokenizer_dir)
    args.data_dir = resolve_path(args.data_dir)
    args.output_dir = resolve_path(args.output_dir)

    if args.folds != 5:
        raise ValueError("BiM6A-FuseNet v1-family experiments require exactly five stratified folds (--folds 5).")
    if args.selection_metric != "ACC":
        raise ValueError("BiM6A-FuseNet v1-family experiments select the best epoch by ACC; use --selection_metric ACC.")
    if args.seed != 42:
        raise ValueError("BiM6A-FuseNet v1-family experiments require --seed 42; fold training seeds are fixed at 42–46.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer.")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be non-negative.")
    if args.max_length < 43:
        raise ValueError("--max_length must be at least 43 for 41 NUC tokens plus CLS/SEP.")
    if args.local_window_radius < 0:
        raise ValueError("--local_window_radius must be non-negative.")
    if args.handcrafted_cnn_channels <= 0:
        raise ValueError("--handcrafted_cnn_channels must be a positive integer.")
    if args.handcrafted_output_dim <= 0:
        raise ValueError("--handcrafted_output_dim must be a positive integer.")
    if args.gated_fusion_dim <= 0:
        raise ValueError("--gated_fusion_dim must be a positive integer.")
    if args.gated_hidden_dim <= 0:
        raise ValueError("--gated_hidden_dim must be a positive integer.")
    args.handcrafted_feature_names = parse_feature_names(args.handcrafted_feature_names)
    if args.use_gated_fusion and (not args.use_film or not args.use_handcrafted_features):
        raise ValueError("--use_gated_fusion requires both --use_film and --use_handcrafted_features.")
    if args.use_projected_concat and (not args.use_film or not args.use_handcrafted_features):
        raise ValueError("--use_projected_concat requires both --use_film and --use_handcrafted_features.")
    if args.use_projected_concat and args.use_gated_fusion:
        raise ValueError("--use_projected_concat cannot be combined with --use_gated_fusion.")
    if args.use_full_mke_eca and not args.use_mke_handcrafted:
        raise ValueError("--use_full_mke_eca requires --use_mke_handcrafted.")
    if args.use_mke_handcrafted:
        if not args.use_film or not args.use_handcrafted_features:
            raise ValueError("--use_mke_handcrafted requires --use_film and --use_handcrafted_features.")
        expected_features = ["onehot", "ncp", "eiip", "enac"]
        if args.handcrafted_feature_names != expected_features:
            raise ValueError(
                "--use_mke_handcrafted requires feature order onehot,ncp,eiip,enac; "
                f"got: {','.join(args.handcrafted_feature_names)}"
            )
        if args.use_projected_concat or args.use_gated_fusion:
            raise ValueError(
                "--use_mke_handcrafted uses --fusion_dim_policy and cannot be combined with "
                "legacy projected-concat or gated-fusion switches."
            )
    if args.handcrafted_only:
        args.use_handcrafted_features = True
        if args.use_gated_fusion:
            raise ValueError("--handcrafted_only cannot be combined with --use_gated_fusion.")
        if args.use_projected_concat:
            raise ValueError("--handcrafted_only cannot be combined with --use_projected_concat.")
        if args.use_mke_handcrafted:
            raise ValueError("--handcrafted_only cannot be combined with --use_mke_handcrafted.")
        if args.use_lora:
            raise ValueError("--handcrafted_only cannot be combined with --use_lora.")
        if args.use_bpe_view:
            raise ValueError("--handcrafted_only cannot be combined with --use_bpe_view.")
        if args.use_film:
            raise ValueError("--handcrafted_only cannot be combined with --use_film.")
    args.cnn_kernel_sizes = parse_cnn_kernel_sizes(args.cnn_kernel_sizes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print(f"model_dir: {args.model_dir}")
    print(f"tokenizer_dir: {args.tokenizer_dir}")
    print(f"data_dir: {args.data_dir}")
    print(f"output_dir: {args.output_dir}")
    print("eval_protocol: strict_cv")
    print(f"selection_metric: {args.selection_metric}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"folds: {args.folds}")
    print(f"freeze_backbone: {args.freeze_backbone}")
    print(f"use_center_pooling: {not args.disable_center_pooling}")
    print(f"use_bpe_view: {args.use_bpe_view}")
    print(f"use_film: {args.use_film}")
    if args.use_film:
        print(f"film_global_view: {args.film_global_view}")
        print(f"local_window_radius: {args.local_window_radius}")
        print(f"film_nuc_pooling: {args.film_nuc_pooling}")
        print(f"cnn_kernel_sizes: {args.cnn_kernel_sizes}")
    print(f"use_lora: {args.use_lora}")
    print(f"use_handcrafted_features: {args.use_handcrafted_features}")
    if args.use_handcrafted_features:
        print(
            "handcrafted_config: "
            f"features={','.join(args.handcrafted_feature_names)}, "
            f"channels={handcrafted_channel_count(args.handcrafted_feature_names)}, "
            f"cnn_channels={args.handcrafted_cnn_channels}, "
            f"output_dim={args.handcrafted_output_dim}, "
            f"handcrafted_only={args.handcrafted_only}, "
            f"use_gated_fusion={args.use_gated_fusion}, "
            f"use_projected_concat={args.use_projected_concat}, "
            f"use_mke_handcrafted={args.use_mke_handcrafted}"
        )
        if args.use_mke_handcrafted:
            print(
                "mke_handcrafted_config: "
                f"full_mke_eca={args.use_full_mke_eca}, fusion_dim_policy={args.fusion_dim_policy}"
            )
        if args.use_gated_fusion:
            print(
                "gated_fusion_config: "
                f"fusion_dim={args.gated_fusion_dim}, hidden_dim={args.gated_hidden_dim}"
            )
    print("checkpoint_policy: delete each temporary fold checkpoint after verified result export")
    if args.use_lora:
        print(
            "lora_config: "
            f"r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
            f"target_modules={args.lora_target_modules}"
        )

    cv_samples, independent_test_samples, data_stats = load_single_dataset_train_test(args.data_dir)
    labels = np.asarray([sample.label for sample in cv_samples])
    min_class_count = min(Counter(labels).values())
    if args.folds > min_class_count:
        raise ValueError(f"--folds={args.folds} is larger than the smallest class count: {min_class_count}")

    tokenizer = load_birna_tokenizer(args.tokenizer_dir, max_length=args.max_length)
    device = select_device()
    print(f"device: {device}")
    print("Data stats:")
    print(json.dumps(json_safe_metrics(data_stats), indent=2, ensure_ascii=False))
    data_audit = audit_dataset(args.data_dir)
    with (args.output_dir / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(data_audit, handle, indent=2, ensure_ascii=False)

    fold_results = []
    indices = np.arange(len(cv_samples))
    split_iter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(indices, labels)
    protocol_folds = [
        (
            fold_idx,
            [cv_samples[int(index)] for index in train_indices],
            [cv_samples[int(index)] for index in val_indices],
        )
        for fold_idx, (train_indices, val_indices) in enumerate(split_iter, start=1)
    ]

    for fold_idx, fold_train, fold_val in protocol_folds:
        print(
            f"Fold {fold_idx}/{args.folds}: "
            f"train={len(fold_train)} benchmark_validation={len(fold_val)} "
            f"independent_test={len(independent_test_samples)}"
        )
        fold_results.append(
            train_one_fold(
                fold_idx=fold_idx,
                train_samples=fold_train,
                val_samples=fold_val,
                independent_test_samples=independent_test_samples,
                tokenizer=tokenizer,
                args=args,
                device=device,
            )
        )

    summary = summarize_folds(fold_results)
    summary["eval_protocol"] = "strict_cv"
    summary["data_stats"] = data_stats
    summary["data_audit"] = data_audit
    summary["args"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with (args.output_dir / "benchmark_cv_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(summary), handle, indent=2, ensure_ascii=False)
    write_benchmark_cv_csv(
        args.output_dir / "benchmark_cv_metrics.csv",
        fold_results,
        summary["benchmark_cv_mean"],
        summary["benchmark_cv_std"],
    )

    ensemble_rows, ensemble_metrics = ensemble_prediction_files(
        [args.output_dir / f"fold_{fold_idx:02d}" / "independent_predictions.csv" for fold_idx in range(1, 6)]
    )
    ensemble_predictions_path = args.output_dir / "independent_ensemble_predictions.csv"
    write_prediction_file(ensemble_predictions_path, ensemble_rows)
    plot_artifacts = save_paper_curves(
        ensemble_rows,
        args.output_dir / "plots",
        model_label=args.model_label,
    )
    ensemble_payload = {
        "method": "five_model_soft_voting",
        "probability_rule": "mean(p_fold_01, ..., p_fold_05)",
        "threshold": 0.5,
        "metrics": ensemble_metrics,
        "predictions": str(ensemble_predictions_path),
        "plots": plot_artifacts,
    }
    with (args.output_dir / "independent_ensemble_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(ensemble_payload), handle, indent=2, ensure_ascii=False)

    print("Five-fold benchmark mean:")
    print(format_metrics(summary["benchmark_cv_mean"]))
    print("Five-fold benchmark sample std:")
    print(format_metrics(summary["benchmark_cv_std"]))
    print("Five-model independent-test soft-voting result:")
    print(format_metrics(ensemble_metrics))
    print(f"Saved CV outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
