from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v0a_birna_nuc_lora",
            "plot_label": "BiRNA-BERT-NUC-LoRA",
            "description": "Pure BiRNA-BERT NUC mean-pooling classifier with Wqkv LoRA.",
        },
        "model": {
            "use_birna_single_branch": True,
            "freeze_backbone": True,
            "use_center_pooling": False,
            "use_bpe_view": False,
            "use_film": False,
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["Wqkv"],
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
            "epochs": 20,
            "batch_size": 32,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "max_length": 64,
            "warmup_ratio": None,
        },
    }
