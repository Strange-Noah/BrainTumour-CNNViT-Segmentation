"""Explicit spatial-unit policy for original UCSD masks and physical metrics."""
import nibabel as nib
import numpy as np

UNKNOWN_UNIT_MASKS = {"BMETS0016_01", "BMETS0095_01"}


def require_mm(image, context="reference"):
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError(f"{context} must declare millimetres")
    zooms = image.header.get_zooms()[:3]
    if len(image.shape) < 3 or not np.isfinite(zooms).all() or any(value <= 0 for value in zooms):
        raise ValueError(f"{context} has invalid spatial dimensions or spacing")
    affine = image.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) <= 0:
        raise ValueError(f"{context} has invalid physical geometry")


def same_grid(image, reference):
    return image.shape == reference.shape and np.allclose(
        image.affine, reference.affine, atol=1e-4, rtol=0
    )


def mask_geometry(mask, reference, exam_id):
    """Interpret an allowlisted missing-unit mask only from an aligned mm T1c."""
    require_mm(reference, "T1c reference")
    if len(mask.shape) != 3 or not same_grid(mask, reference):
        raise ValueError("Mask and T1c reference grids differ")
    unit = mask.header.get_xyzt_units()[0]
    if unit == "mm":
        return []
    if unit == "unknown" and exam_id in UNKNOWN_UNIT_MASKS:
        if not np.allclose(mask.header.get_zooms()[:3], reference.header.get_zooms()[:3], atol=1e-4, rtol=0):
            raise ValueError("Missing-unit mask spacing differs from T1c")
        return ["mask_spatial_units_unknown_interpreted_mm_from_aligned_T1c"]
    raise ValueError("Mask units are undefined or inconsistent with the mm reference")
