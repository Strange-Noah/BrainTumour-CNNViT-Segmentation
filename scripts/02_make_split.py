from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits"
DEFAULT_SEED = 42
EXPECTED_CASES = 484
SPLIT_COUNTS = {"train": 339, "val": 72, "test": 73}


def allocate_counts(group_sizes: dict[int, int], total: int, seed: int) -> dict[int, int]:
    strata = sorted(group_sizes)
    raw = {stratum: total * group_sizes[stratum] / sum(group_sizes.values()) for stratum in strata}
    counts = {stratum: int(np.floor(raw[stratum])) for stratum in strata}
    remaining = total - sum(counts.values())

    rng = np.random.default_rng(seed)
    tie_break = {stratum: rank for rank, stratum in enumerate(rng.permutation(strata))}
    order = sorted(
        strata,
        key=lambda stratum: (raw[stratum] - counts[stratum], -tie_break[stratum]),
        reverse=True,
    )
    for stratum in order[:remaining]:
        counts[stratum] += 1
    return counts


def describe(df: pd.DataFrame, column: str) -> dict[str, float]:
    return {
        "min": float(df[column].min()),
        "q1": float(df[column].quantile(0.25)),
        "median": float(df[column].median()),
        "mean": float(df[column].mean()),
        "q3": float(df[column].quantile(0.75)),
        "max": float(df[column].max()),
    }


def make_split(stats: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    if len(stats) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} patients, got {len(stats)}")

    df = stats.copy()
    df["volume_stratum"] = pd.qcut(df["tumour_volume_mm3"], q=4, labels=False).astype(int)
    stratum_sizes = df["volume_stratum"].value_counts().sort_index().to_dict()
    if set(stratum_sizes.values()) != {121}:
        raise ValueError(f"Expected four strata of 121 patients, got {stratum_sizes}")

    train_counts = allocate_counts(stratum_sizes, SPLIT_COUNTS["train"], seed)
    remaining_counts = {
        stratum: stratum_sizes[stratum] - train_counts[stratum]
        for stratum in stratum_sizes
    }
    val_counts = allocate_counts(remaining_counts, SPLIT_COUNTS["val"], seed + 1)

    split_parts = []
    rng = np.random.default_rng(seed)
    for stratum in sorted(stratum_sizes):
        group = df[df["volume_stratum"] == stratum].copy()
        shuffled_indices = rng.permutation(group.index.to_numpy())

        n_train = train_counts[stratum]
        n_val = val_counts[stratum]
        train_idx = shuffled_indices[:n_train]
        val_idx = shuffled_indices[n_train:n_train + n_val]
        test_idx = shuffled_indices[n_train + n_val:]

        group.loc[train_idx, "split"] = "train"
        group.loc[val_idx, "split"] = "val"
        group.loc[test_idx, "split"] = "test"
        split_parts.append(group)

    split_df = pd.concat(split_parts).sort_values("case_id").reset_index(drop=True)
    split_counts = split_df["split"].value_counts().to_dict()
    for split, expected in SPLIT_COUNTS.items():
        actual = int(split_counts.get(split, 0))
        if actual != expected:
            raise ValueError(f"Expected {expected} {split} patients, got {actual}")

    audit = {
        "seed": seed,
        "split_counts": {split: int(split_counts[split]) for split in ["train", "val", "test"]},
        "volume_quartiles_mm3": {
            "q1": float(df["tumour_volume_mm3"].quantile(0.25)),
            "q2_median": float(df["tumour_volume_mm3"].quantile(0.50)),
            "q3": float(df["tumour_volume_mm3"].quantile(0.75)),
        },
        "stratum_counts": {
            str(stratum): int(count)
            for stratum, count in stratum_sizes.items()
        },
        "split_by_stratum": {
            split: {
                str(stratum): int(count)
                for stratum, count in split_df[split_df["split"] == split]["volume_stratum"]
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            }
            for split in ["train", "val", "test"]
        },
        "balance": {},
    }

    for split in ["train", "val", "test"]:
        part = split_df[split_df["split"] == split]
        audit["balance"][split] = {
            "patients": int(len(part)),
            "tumour_volume_mm3": describe(part, "tumour_volume_mm3"),
            "tumour_volume_ml": {
                key: value / 1000.0
                for key, value in describe(part, "tumour_volume_mm3").items()
            },
            "positive_slice_count": describe(part, "positive_slice_count"),
            "positive_slice_fraction": describe(part, "positive_slice_fraction"),
        }

    return split_df, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    stats_path = args.audit_dir / "patient_tumour_stats.csv"
    manifest_path = args.audit_dir / "train_manifest.csv"
    stats = pd.read_csv(stats_path)
    manifest = pd.read_csv(manifest_path)

    split_input = manifest.merge(
        stats[[
            "case_id",
            "tumour_volume_mm3",
            "positive_slice_count",
            "positive_slice_fraction",
            "tumour_voxels",
        ]],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(split_input) != EXPECTED_CASES:
        raise ValueError(f"Merged manifest has {len(split_input)} patients")

    split_df, audit = make_split(split_input, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_manifest_path = args.output_dir / "frozen_patient_split_seed42.csv"
    audit_path = args.output_dir / "frozen_patient_split_seed42_audit.json"
    preferred_cols = [
        "case_id",
        "image_path",
        "label_path",
        "tumour_volume_mm3",
        "positive_slice_count",
        "positive_slice_fraction",
        "volume_stratum",
        "split",
        "tumour_voxels",
        "voxel_volume_mm3",
        "x",
        "y",
        "z",
        "modalities",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
    ]
    split_df = split_df[[col for col in preferred_cols if col in split_df.columns]]
    split_df.to_csv(frozen_manifest_path, index=False)
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    print(f"Frozen split manifest: {frozen_manifest_path}")
    print(f"Split audit: {audit_path}")
    print(json.dumps(audit["split_counts"], indent=2))
    print("Volume medians mL:")
    for split in ["train", "val", "test"]:
        median_ml = audit["balance"][split]["tumour_volume_ml"]["median"]
        mean_ml = audit["balance"][split]["tumour_volume_ml"]["mean"]
        median_slices = audit["balance"][split]["positive_slice_count"]["median"]
        print(f"  {split}: median={median_ml:.3f}, mean={mean_ml:.3f}, median positive slices={median_slices:.1f}")


if __name__ == "__main__":
    main()
