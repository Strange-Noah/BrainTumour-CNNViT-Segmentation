"""Inspect NIfTI geometry and measure the observed values in a source mask."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def audit(mri_path, mask_path):
    mri, mask = nib.load(mri_path), nib.load(mask_path)
    same_grid = (
        len(mri.shape) == len(mask.shape) == 3
        and mri.shape == mask.shape
        and np.allclose(mri.affine, mask.affine, atol=1e-4, rtol=0)
        and mri.header.get_xyzt_units()[0] == mask.header.get_xyzt_units()[0]
        and np.isfinite(mask.affine).all()
        and abs(np.linalg.det(mask.affine[:3, :3])) > 0
    )
    labels, counts = np.unique(mask.get_fdata(caching="unchanged"), return_counts=True)
    integer_labels = bool(np.isfinite(labels).all() and np.equal(labels, labels.round()).all())
    return {
        "same_grid": bool(same_grid),
        "mri_shape": mri.shape,
        "mask_shape": mask.shape,
        "mask_affine": mask.affine.tolist(),
        "mask_units": mask.header.get_xyzt_units()[0],
        "mask_axcodes": nib.aff2axcodes(mask.affine),
        "integer_labels": integer_labels,
        "labels": dict(zip(labels.astype(int).tolist(), counts.tolist())) if integer_labels else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mri", type=Path)
    parser.add_argument("mask", type=Path)
    args = parser.parse_args()
    result = audit(args.mri, args.mask)
    print(json.dumps(result, indent=2))
    if not result["same_grid"] or not result["integer_labels"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
