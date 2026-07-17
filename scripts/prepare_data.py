#!/usr/bin/env python3
"""Migrate BiRNA_m6A CSV files into semantic benchmark/test names."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DATASETS = {
    "H_b": ("human_brain", "Human_Brain", "human", "brain"),
    "H_k": ("human_kidney", "Human_Kidney", "human", "kidney"),
    "H_l": ("human_liver", "Human_Liver", "human", "liver"),
    "M_b": ("mouse_brain", "Mouse_brain", "mouse", "brain"),
    "M_h": ("mouse_heart", "Mouse_heart", "mouse", "heart"),
    "M_k": ("mouse_kidney", "Mouse_kidney", "mouse", "kidney"),
    "M_l": ("mouse_liver", "Mouse_liver", "mouse", "liver"),
    "M_t": ("mouse_testis", "Mouse_test", "mouse", "testis"),
    "R_b": ("rat_brain", "rat_brain", "rat", "brain"),
    "R_k": ("rat_kidney", "rat_kidney", "rat", "kidney"),
    "R_l": ("rat_liver", "rat_liver", "rat", "liver"),
}

MANIFEST_FIELDS = [
    "dataset_id",
    "canonical_name",
    "source_name",
    "species",
    "tissue",
    "benchmark_path",
    "independent_path",
    "benchmark_positive",
    "benchmark_negative",
    "benchmark_total",
    "independent_positive",
    "independent_negative",
    "independent_total",
    "sequence_length",
    "validation_status",
]


def read_and_validate(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"sequence", "label"} <= set(reader.fieldnames):
            raise ValueError(f"{path} must contain sequence and label columns")
        rows = []
        for index, row in enumerate(reader):
            sequence = row["sequence"].strip().upper()
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"{path}:{index + 2} has invalid label {label}")
            if len(sequence) != 41 or sequence[20] != "A" or not set(sequence) <= set("ACGT"):
                raise ValueError(f"{path}:{index + 2} is not a valid centered 41-nt sequence")
            rows.append({"sequence": sequence, "label": label})
    if not rows:
        raise ValueError(f"{path} is empty")
    counts = Counter(int(row["label"]) for row in rows)
    if counts[0] != counts[1]:
        raise ValueError(f"{path} is not class-balanced: {dict(counts)}")
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "label"])
        writer.writeheader()
        writer.writerows(rows)


def class_counts(rows: list[dict[str, object]]) -> tuple[int, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return counts[1], counts[0]


def migrate(source_root: Path, destination_root: Path) -> None:
    manifest = []
    for dataset_id, (canonical, source_name, species, tissue) in DATASETS.items():
        benchmark = read_and_validate(source_root / source_name / "train.csv")
        independent = read_and_validate(source_root / source_name / "test.csv")
        dataset_dir = destination_root / canonical
        write_rows(dataset_dir / "benchmark.csv", benchmark)
        write_rows(dataset_dir / "independent_test.csv", independent)
        benchmark_positive, benchmark_negative = class_counts(benchmark)
        independent_positive, independent_negative = class_counts(independent)
        manifest.append(
            {
                "dataset_id": dataset_id,
                "canonical_name": canonical,
                "source_name": source_name,
                "species": species,
                "tissue": tissue,
                "benchmark_path": f"{canonical}/benchmark.csv",
                "independent_path": f"{canonical}/independent_test.csv",
                "benchmark_positive": benchmark_positive,
                "benchmark_negative": benchmark_negative,
                "benchmark_total": len(benchmark),
                "independent_positive": independent_positive,
                "independent_negative": independent_negative,
                "independent_total": len(independent),
                "sequence_length": 41,
                "validation_status": "valid",
            }
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    with (destination_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="BiRNA_m6A/data/m6A_41bp")
    parser.add_argument("--destination", type=Path, required=True, help="New semantic data directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    migrate(args.source.resolve(), args.destination.resolve())
    print(f"Migrated {len(DATASETS)} datasets to {args.destination.resolve()}")
