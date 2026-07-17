from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset_utils import SequenceSample, sequence_to_bpe_text, sequence_to_nuc_text
from handcrafted_features import handcrafted_feature_matrix
from metrics_utils import compute_binary_metrics


class RNANucDataset(Dataset):
    def __init__(self, samples: list[SequenceSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "sequence": sample.sequence,
            "label": sample.label,
        }


def build_handcrafted_batch(sequences: list[str], feature_names: list[str] | None = None) -> torch.Tensor:
    features = [handcrafted_feature_matrix(sequence, feature_names=feature_names) for sequence in sequences]
    return torch.tensor(np.stack(features, axis=0), dtype=torch.float32)


class NucDataCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int,
        include_handcrafted: bool = False,
        handcrafted_feature_names: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_handcrafted = include_handcrafted
        self.handcrafted_feature_names = handcrafted_feature_names

    def __call__(self, batch):
        sequences = [item["sequence"] for item in batch]
        texts = [sequence_to_nuc_text(sequence) for sequence in sequences]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded["sequences"] = sequences
        if self.include_handcrafted:
            encoded["handcrafted_features"] = build_handcrafted_batch(sequences, self.handcrafted_feature_names)
        return encoded


class NucViewDataCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int,
        include_handcrafted: bool = False,
        handcrafted_feature_names: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_handcrafted = include_handcrafted
        self.handcrafted_feature_names = handcrafted_feature_names

    def __call__(self, batch):
        sequences = [item["sequence"] for item in batch]
        texts = [sequence_to_nuc_text(sequence) for sequence in sequences]
        nuc_encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        special_token_ids = set(self.tokenizer.all_special_ids)
        nuc_content_mask = (
            nuc_encoded["attention_mask"].bool()
            & ~torch.isin(
                nuc_encoded["input_ids"],
                torch.tensor(sorted(special_token_ids), dtype=nuc_encoded["input_ids"].dtype),
            )
        )
        encoded = {
            f"nuc_{key}": value
            for key, value in nuc_encoded.items()
        }
        encoded["nuc_content_mask"] = nuc_content_mask
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded["sequences"] = sequences
        if self.include_handcrafted:
            encoded["handcrafted_features"] = build_handcrafted_batch(sequences, self.handcrafted_feature_names)
        return encoded


class DualViewDataCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int,
        include_handcrafted: bool = False,
        handcrafted_feature_names: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_handcrafted = include_handcrafted
        self.handcrafted_feature_names = handcrafted_feature_names

    def __call__(self, batch):
        sequences = [item["sequence"] for item in batch]
        nuc_texts = [sequence_to_nuc_text(sequence) for sequence in sequences]
        bpe_texts = [sequence_to_bpe_text(sequence) for sequence in sequences]
        nuc_encoded = self.tokenizer(
            nuc_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        bpe_encoded = self.tokenizer(
            bpe_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        special_token_ids = set(self.tokenizer.all_special_ids)
        nuc_content_mask = (
            nuc_encoded["attention_mask"].bool()
            & ~torch.isin(
                nuc_encoded["input_ids"],
                torch.tensor(sorted(special_token_ids), dtype=nuc_encoded["input_ids"].dtype),
            )
        )
        bpe_content_mask = (
            bpe_encoded["attention_mask"].bool()
            & ~torch.isin(
                bpe_encoded["input_ids"],
                torch.tensor(sorted(special_token_ids), dtype=bpe_encoded["input_ids"].dtype),
            )
        )
        encoded = {
            f"nuc_{key}": value
            for key, value in nuc_encoded.items()
        }
        encoded.update({
            f"bpe_{key}": value
            for key, value in bpe_encoded.items()
        })
        encoded["nuc_content_mask"] = nuc_content_mask
        encoded["bpe_content_mask"] = bpe_content_mask
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded["sequences"] = sequences
        if self.include_handcrafted:
            encoded["handcrafted_features"] = build_handcrafted_batch(sequences, self.handcrafted_feature_names)
        return encoded


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor, list[str]]:
    labels = batch["labels"].to(device)
    sequences = batch["sequences"]
    model_inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if torch.is_tensor(value) and key != "labels"
    }
    return model_inputs, labels, sequences


def train_one_epoch(model, loader, optimizer, criterion, device, epoch: int, freeze_backbone: bool) -> float:
    model.train()
    if freeze_backbone and not getattr(model, "use_lora", False) and hasattr(model, "birna_model"):
        model.birna_model.eval()

    total_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc=f"Epoch {epoch} train", ncols=100)
    for batch in progress:
        model_inputs, labels, _ = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**model_inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str, return_predictions: bool = False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_probs = []
    predictions = []

    for batch in tqdm(loader, desc=desc, ncols=100):
        model_inputs, labels, sequences = move_batch_to_device(batch, device)
        logits = model(**model_inputs)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= 0.5).long()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

        if return_predictions:
            for sequence, label, prob, pred in zip(
                sequences,
                labels.detach().cpu().tolist(),
                probs.detach().cpu().tolist(),
                preds.detach().cpu().tolist(),
            ):
                predictions.append({
                    "sequence": sequence,
                    "label": int(label),
                    "prob": float(prob),
                    "pred": int(pred),
                })

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_binary_metrics(all_labels, all_probs)
    if return_predictions:
        return avg_loss, metrics, predictions
    return avg_loss, metrics


def metric_score(metrics: dict[str, float], selection_metric: str = "ACC") -> float:
    score = metrics.get(selection_metric, math.nan)
    if not math.isnan(score):
        return float(score)
    return -math.inf


def write_train_log_header(path: Path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "ACC",
            "MCC",
            "AUC",
            "AUPRC",
            "F1",
            "Precision",
            "Recall",
            "Sensitivity",
            "Specificity",
            "TP",
            "TN",
            "FP",
            "FN",
            "best_score",
            "is_best",
        ])


def append_train_log(path: Path, row: dict):
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "ACC",
            "MCC",
            "AUC",
            "AUPRC",
            "F1",
            "Precision",
            "Recall",
            "Sensitivity",
            "Specificity",
            "TP",
            "TN",
            "FP",
            "FN",
            "best_score",
            "is_best",
        ])
        writer.writerow(row)


def save_predictions(path: Path, predictions: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence", "label", "prob", "pred"])
        writer.writeheader()
        for sample_id, prediction in enumerate(predictions):
            writer.writerow({"sample_id": sample_id, **prediction})


def delete_temporary_checkpoint(best_model_path: Path) -> bool:
    if best_model_path.exists():
        best_model_path.unlink()
        return True
    return False
