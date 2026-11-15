"""Model-independent review of one MRI visit in voxel coordinates."""

import math

import gradio as gr
import nibabel as nib
import numpy as np

MAX_VOXELS = 256**3
MAX_UPLOAD_MB = 80


def load_volume(path):
    if not str(path).lower().endswith((".nii", ".nii.gz")):
        raise ValueError("Choose a .nii or .nii.gz file.")
    image = nib.load(path)
    if len(image.shape) != 3 or not 0 < math.prod(image.shape) <= MAX_VOXELS:
        raise ValueError("Choose a 3D volume with at most 256³ voxels.")
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(data).all() or not np.isfinite(image.affine).all():
        raise ValueError("The image contains non-finite values or coordinates.")
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("The NIfTI header must specify millimetres.")
    return nib.as_closest_canonical(nib.Nifti1Image(data, image.affine, image.header))


def synthetic_visit():
    grid = np.ogrid[-16:17, -16:17, -12:13]
    x, y, z = grid
    radius = (x / 14) ** 2 + (y / 13) ** 2 + (z / 10) ** 2
    mri = np.where(radius < 1, 1 - radius / 2, 0).astype(np.float32)
    mask = ((x - 3) ** 2 + y**2 + z**2 < 16).astype(np.uint8)
    return mri, mask, np.diag([1.0, 1.0, 2.0, 1.0])


def review(mri_path, mask_path, x, y, z):
    if mri_path:
        mri = load_volume(mri_path)
        data, affine = np.asarray(mri.dataobj), mri.affine
        mask = None
        if mask_path:
            labels = load_volume(mask_path)
            if labels.shape != mri.shape or not np.allclose(labels.affine, affine, atol=1e-4, rtol=0):
                raise ValueError("MRI and mask must share a voxel grid.")
            mask = np.asarray(labels.dataobj)
            if np.any(mask < 0) or np.any(mask != np.floor(mask)) or mask.max() > 65535:
                raise ValueError("Mask values must be integers from 0 to 65535.")
        source = "Uploaded segmentation mask" if mask is not None else "Uploaded MRI"
    else:
        if mask_path:
            raise ValueError("Choose the matching MRI for this mask.")
        data, mask, affine = synthetic_visit()
        source = "Synthetic nonclinical MRI and reference mask"

    voxel_mm3 = abs(np.linalg.det(affine[:3, :3]))
    if not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError("Invalid voxel volume.")
    finite_signal = data[data != 0]
    low, high = np.percentile(finite_signal, [1, 99]) if finite_signal.size else (0, 1)
    width = float(high - low) or 1
    panels = []
    for axis, fraction in enumerate((x, y, z)):
        index = round((data.shape[axis] - 1) * fraction / 100)
        plane = np.take(data, index, axis=axis)
        gray = np.rint(np.clip((plane - low) / width, 0, 1) * 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        if mask is not None:
            rgb[np.take(mask, index, axis=axis) > 0] = (242, 76, 94)
        panels.append(np.flipud(rgb.transpose(1, 0, 2)))

    if mask is not None:
        counts = np.bincount(mask.astype(np.uint16, copy=False).ravel())
        volumes = [(i, count * voxel_mm3 / 1000) for i, count in enumerate(counts) if i and count]
        source += f" · Total {sum(volume for _, volume in volumes):.3f} mL"
        if volumes:
            source += "\n\n" + "\n".join(f"- Label {i} · {volume:.3f} mL" for i, volume in volumes[:32])
            if len(volumes) > 32:
                source += "\n- Showing the first 32 positive labels."
    return (*panels, source)


def render(*args):
    try:
        return review(*args)
    except (ValueError, OSError, nib.filebasedimages.ImageFileError) as error:
        raise gr.Error("Unable to review this volume. Check NIfTI format, units and mask alignment.") from error


with gr.Blocks(delete_cache=(600, 600)) as app:
    gr.Markdown("# Longitudinal MRI Viewer\nReview one visit at a time. Upload only approved, de-identified data.")
    with gr.Row():
        mri = gr.File(label="3D MRI", file_types=[".nii", ".gz"])
        mask = gr.File(label="Aligned segmentation mask", file_types=[".nii", ".gz"])
    with gr.Row():
        sliders = [gr.Slider(0, 100, value=50, label=f"{axis} slice (%)") for axis in ("Sagittal", "Coronal", "Axial")]
    button = gr.Button("Render visit", variant="primary")
    with gr.Row():
        panels = [gr.Image(label=axis, interactive=False) for axis in ("Sagittal", "Coronal", "Axial")]
    summary = gr.Markdown()
    button.click(render, [mri, mask, *sliders], [*panels, summary], api_name="review")
    app.load(lambda: review(None, None, 50, 50, 50), outputs=[*panels, summary], api_visibility="private")

if __name__ == "__main__":
    app.launch(max_file_size=f"{MAX_UPLOAD_MB}mb")
