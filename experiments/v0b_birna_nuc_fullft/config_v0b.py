from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v0b_birna_nuc_fullft",
            "plot_label": "BiRNA-BERT-NUC-FullFT",
            "description": "Pure BiRNA-BERT NUC mean-pooling classifier with full fine-tuning.",
        },
        "model": {
            "use_birna_single_branch": True,
            "freeze_backbone": False,
            "use_center_pooling": False,
            "use_bpe_view": False,
            "use_film": False,
            "use_lora": False,
            "use_handcrafted_features": False,
        },
        "data": {
            "dataset_name": dataset_name,
            "sequence_length": 41,
        },
        "training": {
            "seed": seed,
            "folds": 5,
            "selection_metric": "ACC",
            "epochs": 10,
            "batch_size": 64,
            "lr": 1e-6,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "max_length": 64,
            "warmup_ratio": 0.1,
        },
    }
