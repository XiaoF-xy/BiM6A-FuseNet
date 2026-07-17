"""Reproducible dataset audit retained with every experiment output."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


def _read(path: Path) -> list[tuple[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [(row["sequence"].strip().upper(), int(row["label"])) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    benchmark_path = data_dir / "benchmark.csv"
    independent_path = data_dir / "independent_test.csv"
    benchmark = _read(benchmark_path)
    independent = _read(independent_path)
    for split_name, rows in (("benchmark", benchmark), ("independent_test", independent)):
        if not rows:
            raise ValueError(f"{split_name} dataset is empty: {data_dir}")
        invalid = [
            (sequence, label)
            for sequence, label in rows
            if label not in (0, 1)
            or len(sequence) != 41
            or sequence[20] != "A"
            or not set(sequence) <= set("ACGT")
        ]
        if invalid:
            raise ValueError(f"{split_name} contains invalid 41-nt centered sequences or labels")
        counts = Counter(label for _, label in rows)
        if counts[0] != counts[1]:
            raise ValueError(f"{split_name} is not class-balanced: {dict(counts)}")

    manifest_path = data_dir.parent / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = {row["canonical_name"]: row for row in csv.DictReader(handle)}
    if data_dir.name not in manifest_rows:
        raise ValueError(f"Dataset {data_dir.name} is not registered in {manifest_path}")
    manifest = manifest_rows[data_dir.name]
    expected_benchmark = int(manifest["benchmark_total"])
    expected_independent = int(manifest["independent_total"])
    if len(benchmark) != expected_benchmark or len(independent) != expected_independent:
        raise ValueError(
            "Dataset counts do not match manifest: "
            f"benchmark={len(benchmark)}/{expected_benchmark}, "
            f"independent={len(independent)}/{expected_independent}"
        )
    benchmark_by_sequence: dict[str, set[int]] = defaultdict(set)
    independent_by_sequence: dict[str, set[int]] = defaultdict(set)
    for sequence, label in benchmark:
        benchmark_by_sequence[sequence].add(label)
    for sequence, label in independent:
        independent_by_sequence[sequence].add(label)
    shared_sequences = set(benchmark_by_sequence) & set(independent_by_sequence)
    same_label = sum(
        1 for sequence in shared_sequences if benchmark_by_sequence[sequence] & independent_by_sequence[sequence]
    )
    conflicting_label = sum(
        1
        for sequence in shared_sequences
        if any(
            benchmark_label != independent_label
            for benchmark_label in benchmark_by_sequence[sequence]
            for independent_label in independent_by_sequence[sequence]
        )
    )

    def describe(path: Path, rows: list[tuple[str, int]]) -> dict:
        labels = Counter(label for _, label in rows)
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "total": len(rows),
            "positive": labels[1],
            "negative": labels[0],
            "duplicate_rows": len(rows) - len(set(rows)),
        }

    return {
        "validation_status": "valid",
        "manifest_path": str(manifest_path),
        "benchmark": describe(benchmark_path, benchmark),
        "independent_test": describe(independent_path, independent),
        "cross_split_overlap": {
            "shared_sequences": len(shared_sequences),
            "same_label_sequences": same_label,
            "conflicting_label_sequences": conflicting_label,
            "policy": "reported_not_removed_for_source_comparability",
        },
    }
