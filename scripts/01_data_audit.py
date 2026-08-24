from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import nibabel as nib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TAR = PROJECT_ROOT / "data" / "raw" / "Task01_BrainTumour.tar"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "data_audit"
DEFAULT_RANDOM_SEED = 731

EXPECTED_TAR_SIZE_BYTES = 7_608_266_240
EXPECTED_CASES = 484
EXPECTED_IMAGE_SHAPE = (240, 240, 155, 4)
EXPECTED_LABEL_SHAPE = (240, 240, 155)


def is_clean_member(name: str) -> bool:
    parts = Path(name).parts
    if any(part.startswith("._") or part == "__MACOSX" for part in parts):
        return False
    if name.endswith("dataset.json"):
        return True
    if "/imagesTr/" in name and name.endswith(".nii.gz"):
        return True
    if "/labelsTr/" in name and name.endswith(".nii.gz"):
        return True
    return False


def safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> bool:
    target = destination / member.name
    resolved_target = target.resolve()
    resolved_destination = destination.resolve()
    if not str(resolved_target).startswith(str(resolved_destination)):
        raise ValueError(f"Unsafe tar member path: {member.name}")
    if target.exists() and target.stat().st_size == member.size:
        return False
    tar.extract(member, path=destination, filter="data")
    return True


def real_nii_files(folder: Path) -> list[str]:
    return sorted(
        f for f in os.listdir(folder)
        if f.endswith(".nii.gz") and not f.startswith("._")
    )


def extract_clean_dataset(tar_path: Path, processed_dir: Path) -> tuple[Path, int, dict[str, object]]:
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing raw archive: {tar_path}")

    actual_size = tar_path.stat().st_size
    if actual_size != EXPECTED_TAR_SIZE_BYTES:
        raise ValueError(
            f"Unexpected tar size for {tar_path}: {actual_size}, "
            f"expected {EXPECTED_TAR_SIZE_BYTES}"
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    clean_root = processed_dir / "Task01_BrainTumour"

    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        member_names = [member.name for member in members]
        tar_audit = {
            "tar_path": str(tar_path),
            "tar_size_bytes": actual_size,
            "member_count": len(member_names),
            "first_20_members": member_names[:20],
            "contains_dataset_json": "Task01_BrainTumour/dataset.json" in member_names,
            "contains_imagesTr": any("/imagesTr/" in name for name in member_names),
            "contains_labelsTr": any("/labelsTr/" in name for name in member_names),
        }

        extracted = 0
        for member in members:
            if member.isfile() and is_clean_member(member.name):
                extracted += int(safe_extract_member(tar, member, processed_dir))

    return clean_root, extracted, tar_audit


def build_manifest(clean_root: Path, output_dir: Path) -> pd.DataFrame:
    images_dir = clean_root / "imagesTr"
    labels_dir = clean_root / "labelsTr"

    image_files = real_nii_files(images_dir)
    label_files = real_nii_files(labels_dir)
    image_set = set(image_files)
    label_set = set(label_files)
    missing_labels = sorted(image_set - label_set)
    missing_images = sorted(label_set - image_set)

    if len(image_files) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} real images, got {len(image_files)}")
    if len(label_files) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} real labels, got {len(label_files)}")
    if missing_labels or missing_images:
        raise ValueError(
            f"Pair mismatch. Missing labels: {missing_labels[:10]}; "
            f"missing images: {missing_images[:10]}"
        )

    rows = []
    for image_name in image_files:
        image_path = images_dir / image_name
        label_path = labels_dir / image_name

        img_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))

        if img_nii.shape != EXPECTED_IMAGE_SHAPE:
            raise ValueError(f"{image_name} image shape {img_nii.shape}")
        if label_nii.shape != EXPECTED_LABEL_SHAPE:
            raise ValueError(f"{image_name} label shape {label_nii.shape}")
        if img_nii.shape[:3] != label_nii.shape:
            raise ValueError(f"{image_name} image/label spatial shape mismatch")
        if not np.allclose(img_nii.affine, label_nii.affine):
            raise ValueError(f"{image_name} image/label affine mismatch")

        zooms = tuple(float(z) for z in label_nii.header.get_zooms()[:3])
        rows.append({
            "case_id": image_name.replace(".nii.gz", ""),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "x": img_nii.shape[0],
            "y": img_nii.shape[1],
            "z": img_nii.shape[2],
            "modalities": img_nii.shape[3],
            "spacing_x_mm": zooms[0],
            "spacing_y_mm": zooms[1],
            "spacing_z_mm": zooms[2],
            "voxel_volume_mm3": float(np.prod(zooms)),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "train_manifest.csv", index=False)
    return manifest


def audit_sample(clean_root: Path, manifest: pd.DataFrame) -> dict[str, object]:
    first = manifest.iloc[0]
    img_nii = nib.load(first["image_path"])
    label_nii = nib.load(first["label_path"])
    label = np.asarray(label_nii.dataobj)
    return {
        "case_id": first["case_id"],
        "image_shape": list(img_nii.shape),
        "label_shape": list(label_nii.shape),
        "image_zooms": [float(z) for z in img_nii.header.get_zooms()],
        "label_zooms": [float(z) for z in label_nii.header.get_zooms()],
        "unique_labels": [int(x) for x in np.unique(label)],
        "same_spatial_shape": img_nii.shape[:3] == label_nii.shape,
        "same_affine": bool(np.allclose(img_nii.affine, label_nii.affine)),
        "dataset_json": str(clean_root / "dataset.json"),
    }


def compute_slice_distribution(manifest: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    rows = []
    for row in manifest.itertuples(index=False):
        label_nii = nib.load(row.label_path)
        if label_nii.shape != EXPECTED_LABEL_SHAPE:
            raise ValueError(f"{row.case_id} label shape {label_nii.shape}")

        label = np.asarray(label_nii.dataobj)
        positive_by_slice = np.any(label > 0, axis=(0, 1))
        n_positive = int(np.sum(positive_by_slice))
        n_total = int(label_nii.shape[2])
        n_negative = n_total - n_positive

        rows.append({
            "case_id": row.case_id,
            "label_path": row.label_path,
            "n_total_slices": n_total,
            "n_positive_slices": n_positive,
            "n_negative_slices": n_negative,
            "positive_slice_fraction": n_positive / n_total,
        })

    counts = pd.DataFrame(rows)
    if int(counts["n_total_slices"].nunique()) != 1:
        raise ValueError("Not all cases have the same number of slices")
    if int(counts["n_total_slices"].iloc[0]) != EXPECTED_LABEL_SHAPE[2]:
        raise ValueError("Unexpected number of axial slices")
    if not np.all(counts["n_positive_slices"] + counts["n_negative_slices"] == EXPECTED_LABEL_SHAPE[2]):
        raise ValueError("Positive/negative slice counts do not sum to 155")
    if not np.all(counts["n_positive_slices"] > 0):
        raise ValueError("Found patient(s) with no tumour-positive slices")

    counts.to_csv(output_dir / "tumour_positive_slice_counts.csv", index=False)
    distribution = (
        counts["n_positive_slices"]
        .value_counts()
        .sort_index()
        .rename_axis("n_positive_slices")
        .reset_index(name="num_patients")
    )
    distribution.to_csv(output_dir / "tumour_positive_slice_distribution.csv", index=False)

    return {
        "num_patients": int(len(counts)),
        "total_slices_per_patient": EXPECTED_LABEL_SHAPE[2],
        "total_positive_slices": int(counts["n_positive_slices"].sum()),
        "total_negative_slices": int(counts["n_negative_slices"].sum()),
        "positive_slices_min": int(counts["n_positive_slices"].min()),
        "positive_slices_q1": float(counts["n_positive_slices"].quantile(0.25)),
        "positive_slices_median": float(counts["n_positive_slices"].median()),
        "positive_slices_mean": float(counts["n_positive_slices"].mean()),
        "positive_slices_q3": float(counts["n_positive_slices"].quantile(0.75)),
        "positive_slices_max": int(counts["n_positive_slices"].max()),
        "negative_slices_min": int(counts["n_negative_slices"].min()),
        "negative_slices_median": float(counts["n_negative_slices"].median()),
        "negative_slices_mean": float(counts["n_negative_slices"].mean()),
        "negative_slices_max": int(counts["n_negative_slices"].max()),
        "slice_counts_csv": str(output_dir / "tumour_positive_slice_counts.csv"),
        "distribution_csv": str(output_dir / "tumour_positive_slice_distribution.csv"),
    }


def compute_patient_tumour_stats(manifest: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for row in manifest.itertuples(index=False):
        label_nii = nib.load(row.label_path)
        if label_nii.shape != EXPECTED_LABEL_SHAPE:
            raise ValueError(f"{row.case_id} label shape {label_nii.shape}")

        voxel_volume_mm3 = float(row.voxel_volume_mm3)
        label = np.asarray(label_nii.dataobj)
        positive_by_slice = np.any(label > 0, axis=(0, 1))
        tumour_voxels = int(np.sum(label > 0))
        tumour_volume_mm3 = float(tumour_voxels * voxel_volume_mm3)
        positive_slice_count = int(np.sum(positive_by_slice))
        positive_slice_fraction = positive_slice_count / int(label_nii.shape[2])

        rows.append({
            "case_id": row.case_id,
            "tumour_volume_mm3": tumour_volume_mm3,
            "positive_slice_count": positive_slice_count,
            "positive_slice_fraction": positive_slice_fraction,
            "tumour_voxels": tumour_voxels,
            "voxel_volume_mm3": voxel_volume_mm3,
        })

    stats = pd.DataFrame(rows)
    if len(stats) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} patient stats rows, got {len(stats)}")
    if not np.all(stats["tumour_volume_mm3"] > 0):
        raise ValueError("Found patient(s) with zero tumour volume")

    stats_csv = output_dir / "patient_tumour_stats.csv"
    stats.to_csv(stats_csv, index=False)

    volume_distribution = (
        stats["tumour_volume_mm3"]
        .round()
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("tumour_volume_mm3_rounded")
        .reset_index(name="num_patients")
    )
    volume_distribution_csv = output_dir / "tumour_volume_distribution.csv"
    volume_distribution.to_csv(volume_distribution_csv, index=False)

    summary = {
        "num_patients": int(len(stats)),
        "patient_tumour_stats_csv": str(stats_csv),
        "volume_distribution_csv": str(volume_distribution_csv),
        "tumour_volume_mm3_min": float(stats["tumour_volume_mm3"].min()),
        "tumour_volume_mm3_q1": float(stats["tumour_volume_mm3"].quantile(0.25)),
        "tumour_volume_mm3_median": float(stats["tumour_volume_mm3"].median()),
        "tumour_volume_mm3_mean": float(stats["tumour_volume_mm3"].mean()),
        "tumour_volume_mm3_q3": float(stats["tumour_volume_mm3"].quantile(0.75)),
        "tumour_volume_mm3_max": float(stats["tumour_volume_mm3"].max()),
        "tumour_volume_ml_min": float(stats["tumour_volume_mm3"].min() / 1000.0),
        "tumour_volume_ml_median": float(stats["tumour_volume_mm3"].median() / 1000.0),
        "tumour_volume_ml_mean": float(stats["tumour_volume_mm3"].mean() / 1000.0),
        "tumour_volume_ml_max": float(stats["tumour_volume_mm3"].max() / 1000.0),
    }
    return stats, summary


def save_random_sample_visualizations(
    manifest: pd.DataFrame,
    output_dir: Path,
    seed: int,
    n_samples: int = 3,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    sample_dir = output_dir / "sample_visualizations"
    sample_dir.mkdir(parents=True, exist_ok=True)

    indices = sorted(rng.choice(len(manifest), size=n_samples, replace=False).tolist())
    outputs = []
    for idx in indices:
        row = manifest.iloc[idx]
        img_nii = nib.load(row["image_path"])
        label_nii = nib.load(row["label_path"])
        image = np.asarray(img_nii.dataobj)
        label = np.asarray(label_nii.dataobj)

        tumour_pixels_per_slice = np.sum(label > 0, axis=(0, 1))
        slice_idx = int(np.argmax(tumour_pixels_per_slice))
        tumour_pixels = int(tumour_pixels_per_slice[slice_idx])

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for modality in range(4):
            axes[modality].imshow(
                image[:, :, slice_idx, modality].T,
                cmap="gray",
                origin="lower",
            )
            axes[modality].set_title(f"{row['case_id']} M{modality}")
            axes[modality].axis("off")

        axes[4].imshow(label[:, :, slice_idx].T, origin="lower")
        axes[4].set_title(f"Label z={slice_idx}")
        axes[4].axis("off")
        fig.suptitle(f"{row['case_id']} | tumour pixels on slice: {tumour_pixels}")
        plt.tight_layout()

        out_path = sample_dir / f"{row['case_id']}_slice_{slice_idx:03d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        outputs.append({
            "case_id": row["case_id"],
            "slice_idx": slice_idx,
            "tumour_pixels": tumour_pixels,
            "path": str(out_path),
        })

    with open(output_dir / "sample_visualizations.json", "w") as f:
        json.dump({
            "seed": seed,
            "n_samples": n_samples,
            "samples": outputs,
        }, f, indent=2)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", type=Path, default=DEFAULT_TAR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_root, extracted, tar_audit = extract_clean_dataset(args.tar, args.processed_dir)
    manifest = build_manifest(clean_root, args.output_dir)
    sample_audit = audit_sample(clean_root, manifest)
    slice_summary = compute_slice_distribution(manifest, args.output_dir)
    patient_stats, volume_summary = compute_patient_tumour_stats(manifest, args.output_dir)
    sample_visualizations = save_random_sample_visualizations(manifest, args.output_dir, args.seed)

    filter_summary = {
        "source_archive": str(args.tar),
        "clean_data_root": str(clean_root),
        "output_dir": str(args.output_dir),
        "manifest_csv": str(args.output_dir / "train_manifest.csv"),
        "num_cases": int(len(manifest)),
        "newly_extracted_files": extracted,
        "tar_audit": tar_audit,
        "sample_audit": sample_audit,
        "sample_visualizations": sample_visualizations,
        "excluded_rule": "Extract dataset.json plus imagesTr/labelsTr *.nii.gz; exclude path parts starting with ._ and __MACOSX.",
        "expected_image_shape": list(EXPECTED_IMAGE_SHAPE),
        "expected_label_shape": list(EXPECTED_LABEL_SHAPE),
        "labels": {
            "0": "background",
            "1": "edema",
            "2": "non-enhancing tumor",
            "3": "enhancing tumour",
        },
        "volume_note": "With 1 mm^3 voxels, voxel counts equal mm^3; divide by 1000 for mL.",
    }

    with open(args.output_dir / "filter_summary.json", "w") as f:
        json.dump(filter_summary, f, indent=2)
    with open(args.output_dir / "tumour_positive_slice_summary.json", "w") as f:
        json.dump(slice_summary, f, indent=2)
    with open(args.output_dir / "tumour_volume_summary.json", "w") as f:
        json.dump(volume_summary, f, indent=2)

    print(f"Clean data root: {clean_root}")
    print(f"Cases: {len(manifest)}")
    print(f"Newly extracted files: {extracted}")
    print(f"Manifest: {args.output_dir / 'train_manifest.csv'}")
    print(f"Slice counts: {args.output_dir / 'tumour_positive_slice_counts.csv'}")
    print(f"Slice distribution: {args.output_dir / 'tumour_positive_slice_distribution.csv'}")
    print(f"Patient tumour stats: {args.output_dir / 'patient_tumour_stats.csv'}")
    print(f"Volume distribution: {args.output_dir / 'tumour_volume_distribution.csv'}")
    print(f"Sample visualizations: {args.output_dir / 'sample_visualizations'}")
    print(f"Total tumour-positive slices: {slice_summary['total_positive_slices']}")
    print(f"Total tumour-negative slices: {slice_summary['total_negative_slices']}")
    print(f"Median tumour volume mm3: {volume_summary['tumour_volume_mm3_median']}")
    print(f"Mean tumour volume mm3: {volume_summary['tumour_volume_mm3_mean']}")


if __name__ == "__main__":
    main()
