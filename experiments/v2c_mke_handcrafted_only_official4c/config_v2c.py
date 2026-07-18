from __future__ import annotations


def get_overrides(dataset_name: str, seed: int) -> dict:
    return {
        "experiment": {
            "version_name": "v2c_mke_handcrafted_only_official4c",
            "plot_label": "MKE-ResNet-official4c",
            "description": (
                "Pure handcrafted public MKE-ResNet architecture with the repository's "
                "four-channel chemical representation and strict five-fold evaluation."
            ),
        },
        "model": {
            "freeze_backbone": False,
            "use_center_pooling": False,
            "use_bpe_view": False,
            "use_film": False,
            "use_lora": False,
            "use_handcrafted_features": True,
            "handcrafted_feature_names": ["onehot", "chemical4", "eiip", "enac"],
            "handcrafted_only": True,
            "use_mke_handcrafted": False,
            "use_full_mke_eca": False,
            "use_official_mke_handcrafted": True,
        },
        "data": {
            "dataset_name": dataset_name,
            "sequence_length": 41,
        },
        "training": {
            "seed": seed,
            "epochs": 100,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "folds": 5,
            "max_length": 64,
            "optimizer": "adam",
            "scheduler_patience": 10,
            "scheduler_factor": 0.1,
            "early_stopping_patience": 20,
        },
    }
