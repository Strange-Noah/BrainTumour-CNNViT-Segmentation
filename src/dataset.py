from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image_path: Path
    label_path: Path
    voxel_volume_mm3: float


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"case_id", "image_path", "label_path", "voxel_volume_mm3"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return manifest


def iter_cases(manifest: pd.DataFrame):
    for row in manifest.itertuples(index=False):
        yield CaseRecord(
            case_id=str(row.case_id),
            image_path=Path(row.image_path),
            label_path=Path(row.label_path),
            voxel_volume_mm3=float(row.voxel_volume_mm3),
        )


def positive_slice_mask(label_path: str | Path) -> np.ndarray:
    label_nii = nib.load(str(label_path))
    if label_nii.shape != (240, 240, 155):
        raise ValueError(f"Unexpected label shape for {label_path}: {label_nii.shape}")
    label = np.asarray(label_nii.dataobj)
    return np.any(label > 0, axis=(0, 1))


def tumour_volume_ml(label_path: str | Path, voxel_volume_mm3: float = 1.0) -> dict[str, float]:
    label = np.asarray(nib.load(str(label_path)).dataobj)
    whole = float(np.sum(label > 0) * voxel_volume_mm3 / 1000.0)
    edema = float(np.sum(label == 1) * voxel_volume_mm3 / 1000.0)
    non_enhancing = float(np.sum(label == 2) * voxel_volume_mm3 / 1000.0)
    enhancing = float(np.sum(label == 3) * voxel_volume_mm3 / 1000.0)
    return {
        "whole_tumour_ml": whole,
        "edema_ml": edema,
        "non_enhancing_tumour_ml": non_enhancing,
        "enhancing_tumour_ml": enhancing,
    }

