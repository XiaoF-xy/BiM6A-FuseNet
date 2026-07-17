from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SEQ_COLUMNS = {"sequence", "seq", "rna", "dna", "fragment"}
LABEL_COLUMNS = {"label", "labels", "target", "class", "y", "is_m6a", "m6a"}
VALID_SUFFIXES = {".csv", ".tsv", ".txt", ".fa", ".fasta", ".fna"}


@dataclass(frozen=True)
class SequenceSample:
    sequence: str
    label: int
    source: str = ""


@dataclass
class DatasetSplits:
    train: list[SequenceSample]
    val: list[SequenceSample]
    test: list[SequenceSample]
    stats: dict


def clean_sequence(raw_sequence: str, expected_length: int = 41) -> tuple[str | None, str | None]:
    sequence = re.sub(r"\s+", "", str(raw_sequence).upper()).replace("U", "T")
    if not sequence:
        return None, "empty_sequence"
    if re.search(r"[^ACGT]", sequence):
        return None, "invalid_base"
    if len(sequence) != expected_length:
        return None, "bad_length"
    return sequence, None


def sequence_to_nuc_text(sequence: str) -> str:
    return " ".join(sequence)


def sequence_to_bpe_text(sequence: str) -> str:
    return sequence


def infer_split_from_path(path: Path) -> str | None:
    text = " ".join(part.lower() for part in path.parts)
    name = path.stem.lower()
    if _has_split_token(name, text, ("train", "training")):
        return "train"
    if _has_split_token(name, text, ("val", "valid", "validation", "dev")):
        return "val"
    if _has_split_token(name, text, ("test", "testing")):
        return "test"
    return None


def infer_label_from_path(path: Path) -> int | None:
    text = " ".join(part.lower() for part in path.parts)
    if re.search(r"(^|[^a-z0-9])(positive|pos)([^a-z0-9]|$)", text):
        return 1
    if re.search(r"(^|[^a-z0-9])(negative|neg)([^a-z0-9]|$)", text):
        return 0
    return None


def parse_label(value: object, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "t", "yes", "y", "positive", "pos", "m6a"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "negative", "neg", "non_m6a", "non-m6a"}:
        return 0
    try:
        number = float(text)
    except ValueError:
        return default
    if number == 1:
        return 1
    if number == 0:
        return 0
    return default


def discover_data_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

    files = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.stem.lower() == "manifest":
            continue
        if path.suffix.lower() in VALID_SUFFIXES:
            files.append(path)
    return files


def select_data_files(data_dir: Path) -> list[Path]:
    files = discover_data_files(data_dir)
    if not files:
        raise ValueError(f"No supported data files found under: {data_dir}")

    all_split_files = [
        path for path in files
        if path.parent == data_dir and path.stem.lower() in {
            "all_train", "all_test", "all_val", "all_valid", "all_dev"
        }
    ]
    all_splits = {infer_split_from_path(path) for path in all_split_files}
    if {"train", "test"}.issubset(all_splits):
        return sorted(all_split_files)

    root_split_files = [path for path in files if path.parent == data_dir and infer_split_from_path(path)]
    root_splits = {infer_split_from_path(path) for path in root_split_files}
    if {"train", "test"}.issubset(root_splits):
        return sorted(root_split_files)

    return files


def load_dataset_splits(
    data_dir: Path,
    seed: int = 42,
    expected_length: int = 41,
) -> DatasetSplits:
    data_dir = Path(data_dir)
    selected_files = select_data_files(data_dir)
    samples_by_split: dict[str, list[SequenceSample]] = defaultdict(list)
    unsplit_samples: list[SequenceSample] = []
    skipped = Counter()
    raw_records = 0

    for path in selected_files:
        split = infer_split_from_path(path)
        file_samples, file_stats = read_samples_from_file(path, expected_length=expected_length)
        raw_records += file_stats["raw_records"]
        skipped.update(file_stats["skipped"])
        if split is None:
            unsplit_samples.extend(file_samples)
        else:
            samples_by_split[split].extend(file_samples)

    has_original_split = bool(samples_by_split["train"]) and bool(samples_by_split["test"])
    if has_original_split:
        train = samples_by_split["train"]
        val = samples_by_split["val"]
        test = samples_by_split["test"]
        if unsplit_samples:
            skipped["ignored_unsplit_with_existing_train_test"] += len(unsplit_samples)
        if not val:
            train, val = stratified_train_val_split(train, val_ratio=0.1, seed=seed)
    else:
        all_samples = []
        for split_samples in samples_by_split.values():
            all_samples.extend(split_samples)
        all_samples.extend(unsplit_samples)
        train, val, test = stratified_three_way_split(all_samples, seed=seed)

    if not train:
        raise ValueError(f"No training samples were loaded from: {data_dir}")
    if not val:
        raise ValueError(f"No validation samples were loaded from: {data_dir}")
    if not test:
        raise ValueError(f"No test samples were loaded from: {data_dir}")

    stats = {
        "data_dir": str(data_dir),
        "used_files": [str(path) for path in selected_files],
        "raw_records": raw_records,
        "kept_records": len(train) + len(val) + len(test),
        "skipped": dict(skipped),
        "original_train_test_split": has_original_split,
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "label_counts": {
            "train": dict(Counter(sample.label for sample in train)),
            "val": dict(Counter(sample.label for sample in val)),
            "test": dict(Counter(sample.label for sample in test)),
        },
    }
    return DatasetSplits(train=train, val=val, test=test, stats=stats)


def read_samples_from_file(path: Path, expected_length: int = 41) -> tuple[list[SequenceSample], dict]:
    path = Path(path)
    skipped = Counter()
    raw_records = 0
    default_label = infer_label_from_path(path)

    if path.suffix.lower() in {".fa", ".fasta", ".fna"}:
        records = _read_fasta_records(path)
    else:
        records = _read_table_or_line_records(path)

    samples: list[SequenceSample] = []
    for raw_sequence, raw_label in records:
        raw_records += 1
        label = parse_label(raw_label, default=default_label)
        if label is None:
            skipped["missing_label"] += 1
            continue
        sequence, reason = clean_sequence(raw_sequence, expected_length=expected_length)
        if reason is not None:
            skipped[reason] += 1
            continue
        samples.append(SequenceSample(sequence=sequence, label=label, source=str(path)))

    return samples, {"raw_records": raw_records, "skipped": dict(skipped)}


def stratified_train_val_split(
    samples: list[SequenceSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[SequenceSample], list[SequenceSample]]:
    train, val = _stratified_split(samples, ratios=(1.0 - val_ratio, val_ratio), seed=seed)
    return train, val


def stratified_three_way_split(
    samples: list[SequenceSample],
    seed: int,
) -> tuple[list[SequenceSample], list[SequenceSample], list[SequenceSample]]:
    train, val, test = _stratified_split(samples, ratios=(0.8, 0.1, 0.1), seed=seed)
    return train, val, test


def _read_fasta_records(path: Path) -> list[tuple[str, object]]:
    records = []
    header = None
    sequence_parts = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(("".join(sequence_parts), _infer_label_from_header(header)))
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)
        if header is not None:
            records.append(("".join(sequence_parts), _infer_label_from_header(header)))
    return records


def _read_table_or_line_records(path: Path) -> list[tuple[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    if path.suffix.lower() == ".txt" and "\t" in lines[0]:
        delimiter = "\t"
    elif path.suffix.lower() == ".txt" and "," not in lines[0]:
        return [(line.strip(), None) for line in lines if not line.lstrip().startswith("#")]

    reader = csv.reader(lines, delimiter=delimiter)
    rows = [row for row in reader if row]
    if not rows:
        return []

    header = [cell.strip().lower() for cell in rows[0]]
    has_header = any(cell in SEQ_COLUMNS or cell in LABEL_COLUMNS for cell in header)
    if has_header:
        seq_idx = _first_index(header, SEQ_COLUMNS)
        label_idx = _first_index(header, LABEL_COLUMNS)
        data_rows = rows[1:]
    else:
        seq_idx = _infer_sequence_column(rows[0])
        label_idx = 1 if len(rows[0]) > 1 else None
        data_rows = rows

    if seq_idx is None:
        return []

    records = []
    for row in data_rows:
        if seq_idx >= len(row):
            continue
        raw_sequence = row[seq_idx].strip()
        raw_label = row[label_idx].strip() if label_idx is not None and label_idx < len(row) else None
        records.append((raw_sequence, raw_label))
    return records


def _stratified_split(
    samples: list[SequenceSample],
    ratios: tuple[float, ...],
    seed: int,
) -> tuple[list[SequenceSample], ...]:
    if not samples:
        return tuple([] for _ in ratios)

    rng = random.Random(seed)
    buckets: dict[int, list[SequenceSample]] = defaultdict(list)
    for sample in samples:
        buckets[sample.label].append(sample)

    splits = [[] for _ in ratios]
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        boundaries = []
        used = 0
        for ratio in ratios[:-1]:
            count = int(round(n * ratio))
            count = min(count, n - used)
            boundaries.append(count)
            used += count
        boundaries.append(n - used)

        start = 0
        for split, count in zip(splits, boundaries):
            split.extend(bucket[start:start + count])
            start += count

    for split in splits:
        rng.shuffle(split)
    return tuple(splits)


def _first_index(values: list[str], choices: set[str]) -> int | None:
    for idx, value in enumerate(values):
        if value in choices:
            return idx
    return None


def _infer_sequence_column(row: list[str]) -> int | None:
    for idx, value in enumerate(row):
        compact = re.sub(r"\s+", "", value.upper()).replace("U", "T")
        if compact and re.fullmatch(r"[ACGTN]+", compact):
            return idx
    return 0 if row else None


def _infer_label_from_header(header: str) -> int | None:
    parsed = parse_label(header, default=None)
    if parsed is not None:
        return parsed
    return infer_label_from_text(header)


def infer_label_from_text(text: str) -> int | None:
    lowered = text.lower()
    if re.search(r"(^|[^a-z0-9])(positive|pos|label[:=_-]?1|class[:=_-]?1)([^a-z0-9]|$)", lowered):
        return 1
    if re.search(r"(^|[^a-z0-9])(negative|neg|label[:=_-]?0|class[:=_-]?0)([^a-z0-9]|$)", lowered):
        return 0
    return None


def _has_split_token(name: str, text: str, tokens: Iterable[str]) -> bool:
    for token in tokens:
        pattern = rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)"
        if re.search(pattern, name) or re.search(pattern, text):
            return True
    return False
