from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v1_baseline",
            "description": (
                "BiM6A-FuseNet v1 baseline migrated without architectural changes "
                "from the BiRNA_m6A v9a model."
            ),
        },
        "model": {
            "use_center_pooling": False,
            "use_bpe_view": False,
            "use_film": True,
            "film_global_view": "nuc",
            "film_nuc_pooling": "center_cnn_mean",
            "local_window_radius": 3,
            "cnn_kernel_sizes": [3, 5, 7],
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["Wqkv"],
            "freeze_backbone": True,
            "use_handcrafted_features": True,
            "handcrafted_feature_names": ["onehot", "ncp", "eiip", "enac"],
            "handcrafted_cnn_channels": 64,
            "handcrafted_output_dim": 128,
        },
        "data": {
            "dataset_name": dataset_name,
            "sequence_length": 41,
        },
        "training": {
            "seed": seed,
            "epochs": 20,
            "batch_size": 32,
            "lr": 1e-4,
            "folds": 5,
            "max_length": 64,
        },
    }
