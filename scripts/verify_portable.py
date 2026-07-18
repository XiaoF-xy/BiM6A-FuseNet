#!/usr/bin/env python3
"""Verify that the local folder contains every file required on the server."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHT = ROOT / "pretrained" / "birna-bert-model" / "pytorch_model.bin"
EXPECTED_WEIGHT_SHA256 = "4833ca3207d1908a86acffc84d6435379ab65c8da8f1790065c3c683bdacef3b"
REQUIRED = [
    WEIGHT,
    ROOT / "pretrained" / "birna-bert-model" / "config.json",
    ROOT / "pretrained" / "birna-bert-model" / "tokenizer.json",
    ROOT / "data" / "m6a_41nt" / "manifest.csv",
    ROOT / "experiments" / "v0a_birna_nuc_lora" / "config_v0a.py",
    ROOT / "experiments" / "v0b_birna_nuc_fullft" / "config_v0b.py",
    ROOT / "experiments" / "v1_baseline" / "config_v1.py",
    ROOT / "experiments" / "mke_variants_common.py",
    ROOT / "experiments" / "v2a_mke_res_eca_native" / "config_v2a.py",
    ROOT / "experiments" / "v2b_mke_res_eca_proj256" / "config_v2b.py",
    ROOT / "experiments" / "v2c_mke_handcrafted_only_official4c" / "config_v2c.py",
    ROOT / "experiments" / "v3a_full_mke_eca_native" / "config_v3a.py",
    ROOT / "experiments" / "v3b_full_mke_eca_proj256" / "config_v3b.py",
    ROOT / "src" / "model_birna_mke.py",
    ROOT / "src" / "model_birna_single.py",
    ROOT / "src" / "model_mke_handcrafted.py",
    ROOT / "src" / "mke_official_features.py",
    ROOT / "src" / "model_mke_official.py",
    ROOT / "src" / "training_control.py",
    ROOT / "src" / "train_cv.py",
    ROOT / "train.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Portable folder is incomplete; missing: {', '.join(missing)}")
    actual_hash = sha256(WEIGHT)
    if actual_hash != EXPECTED_WEIGHT_SHA256:
        raise ValueError(f"BiRNA-BERT weight checksum mismatch: {actual_hash}")
    print(f"project_root: {ROOT}")
    print(f"weight_sha256: {actual_hash}")
    print("portable_check: OK")


if __name__ == "__main__":
    main()
