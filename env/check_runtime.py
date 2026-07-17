from __future__ import annotations

import sys
from pathlib import Path

import peft
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset_utils import read_samples_from_file  # noqa: E402
from handcrafted_features import handcrafted_channel_count  # noqa: E402
from model_birna_film import BiRNAFiLMHandcraftedClassifier  # noqa: E402
from model_birna_nuc import load_birna_tokenizer  # noqa: E402
from training_utils import NucViewDataCollator  # noqa: E402


def main():
    model_dir = PROJECT_DIR / "pretrained" / "birna-bert-model"
    data_path = PROJECT_DIR / "data" / "m6a_41nt" / "human_brain" / "benchmark.csv"

    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"peft: {peft.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")

    samples, stats = read_samples_from_file(data_path, expected_length=41)
    print(f"benchmark_records: {len(samples)}; skipped: {stats['skipped']}")
    tokenizer = load_birna_tokenizer(model_dir, max_length=64)
    collator = NucViewDataCollator(
        tokenizer=tokenizer,
        max_length=64,
        include_handcrafted=True,
        handcrafted_feature_names=["onehot", "ncp", "eiip", "enac"],
    )
    batch = collator([{"sequence": sample.sequence, "label": sample.label} for sample in samples[:2]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiRNAFiLMHandcraftedClassifier(
        model_dir=model_dir,
        freeze_backbone=True,
        film_global_view="nuc",
        local_window_radius=3,
        film_nuc_pooling="center_cnn_mean",
        cnn_kernel_sizes=[3, 5, 7],
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=["Wqkv"],
        handcrafted_input_channels=handcrafted_channel_count(["onehot", "ncp", "eiip", "enac"]),
        handcrafted_cnn_channels=64,
        handcrafted_output_dim=128,
    ).to(device)
    model.eval()
    model_inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if torch.is_tensor(value) and key != "labels"
    }
    with torch.no_grad():
        logits = model(**model_inputs)
    print(f"v1_logits_shape: {tuple(logits.shape)}")
    if tuple(logits.shape) != (2, 2):
        raise RuntimeError(f"Unexpected v1 output shape: {tuple(logits.shape)}")
    print("runtime_check: OK")


if __name__ == "__main__":
    main()
