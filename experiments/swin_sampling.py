"""Sampling and crop utilities for controlled Swin UNETR BMETS experiments."""
from collections import defaultdict
import hashlib
import os
from pathlib import Path

import numpy as np
from scipy import ndimage


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


def component_index(target, affine):
    """Return 26-connected component labels and physical volumes for a binary mask."""
    labels, count = ndimage.label(np.asarray(target, dtype=bool), structure=CONNECTIVITY_26)
    voxel_volume_mm3 = abs(float(np.linalg.det(np.asarray(affine)[:3, :3])))
    counts = np.bincount(labels.ravel(), minlength=count + 1)
    objects = ndimage.find_objects(labels)
    components = {}
    for component_id, region in enumerate(objects, start=1):
        if region is None:
            continue
        components[component_id] = {
            "volume_mm3": float(counts[component_id] * voxel_volume_mm3),
            "bbox": tuple((int(axis.start), int(axis.stop)) for axis in region),
        }
    return labels.astype(np.int32, copy=False), components


def acquisition_support(images):
    """MRI-derived acquisition support; this is not a skull-stripped brain mask."""
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[0] != 4:
        raise ValueError(f"Expected 4 x X x Y x Z MRI data, got {images.shape}")
    support = np.any(np.isfinite(images) & (images != 0), axis=0)
    if not support.any():
        raise ValueError("Cannot estimate MRI support: all four modalities are zero")
    # Remove isolated nonzero islands while retaining the dominant head volume.
    labels, count = ndimage.label(support, structure=CONNECTIVITY_26)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    return labels == int(sizes.argmax())


def cached_acquisition_support(images, cache_dir=None, cache_key=None):
    """Build or reuse the MRI support mask without relabelling volumes every epoch."""
    images = np.asarray(images)
    if cache_dir is None or cache_key is None:
        return acquisition_support(images)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()
    path = cache_dir / f"{key}.npz"
    if path.exists():
        with np.load(path) as stored:
            support = stored["support"].astype(bool, copy=False)
        if support.shape != images.shape[-3:]:
            raise ValueError(f"Cached MRI support shape mismatch for {cache_key}")
        return support

    support = acquisition_support(images)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, support=support.astype(np.uint8))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return support


def acquisition_bounds(images):
    """Return low/high voxel bounds of nonzero MRI support without a 3D label map."""
    images = np.asarray(images)
    union = np.any(images != 0, axis=0)
    occupied = (
        union.any(axis=(1, 2)),
        union.any(axis=(0, 2)),
        union.any(axis=(0, 1)),
    )
    bounds = []
    for axis, mask in enumerate(occupied):
        locations = np.flatnonzero(mask)
        if not len(locations):
            bounds.append((0, images.shape[axis + 1]))
        else:
            bounds.append((int(locations[0]), int(locations[-1]) + 1))
    return tuple(bounds)


def patient_balanced_draw_counts(visits, indices, rng, epoch):
    """Allocate the same number of draws to every patient, then rotate their visits."""
    by_patient = defaultdict(list)
    for index in indices:
        by_patient[visits[index]["patient_id"]].append(index)
    patients = sorted(by_patient)
    if not patients:
        raise ValueError("No visits available for patient-balanced sampling")

    steps = len(indices)
    per_patient, remainder = divmod(steps, len(patients))
    extra = set(rng.sample(patients, remainder)) if remainder else set()
    counts = defaultdict(int)
    for patient in patients:
        patient_visits = sorted(by_patient[patient])
        draws = per_patient + int(patient in extra)
        # Rotate starting examinations so patients with fewer visits do not always
        # receive their extra crop from the same examination.
        offset = (epoch - 1) % len(patient_visits)
        for draw in range(draws):
            index = patient_visits[(offset + draw) % len(patient_visits)]
            counts[index] += 1
    return dict(counts)


def _crop_pad(array, center, shape, pad_value=0):
    """Crop the last three dimensions around center, padding at image boundaries."""
    spatial = array.shape[-3:]
    starts = [int(center[d]) - shape[d] // 2 for d in range(3)]
    stops = [starts[d] + shape[d] for d in range(3)]
    source_starts = [max(0, starts[d]) for d in range(3)]
    source_stops = [min(spatial[d], stops[d]) for d in range(3)]
    slices = tuple(slice(source_starts[d], source_stops[d]) for d in range(3))
    patch = array[(..., *slices)]
    pads = [(0, 0)] * (array.ndim - 3) + [
        (source_starts[d] - starts[d], stops[d] - source_stops[d])
        for d in range(3)
    ]
    return np.pad(patch, pads, mode="constant", constant_values=pad_value)


def _resize_exact(array, roi, order):
    """Resize only spatial axes and correct rounding to an exact cubic ROI."""
    factors = tuple(roi / size for size in array.shape[-3:])
    resized = ndimage.zoom(
        array,
        (1.0,) * (array.ndim - 3) + factors,
        order=order,
        mode="nearest" if order else "nearest",
        prefilter=bool(order),
    )
    desired = (roi,) * 3
    starts = [max(0, (resized.shape[-3 + d] - roi) // 2) for d in range(3)]
    slices = tuple(slice(starts[d], starts[d] + min(roi, resized.shape[-3 + d])) for d in range(3))
    resized = resized[(..., *slices)]
    pads = [(0, 0)] * (resized.ndim - 3) + [
        (0, max(0, desired[d] - resized.shape[-3 + d])) for d in range(3)
    ]
    return np.pad(resized, pads, mode="constant", constant_values=0)


def _center_for_bbox(component, center, crop_shape):
    """Move jittered center only as much as needed to retain a selected component."""
    adjusted = []
    for axis, (low, high) in enumerate(component["bbox"]):
        half = crop_shape[axis] // 2
        lower_center = high - half
        upper_center = low + (crop_shape[axis] - half)
        if lower_center <= upper_center:
            adjusted.append(min(max(int(center[axis]), lower_center), upper_center))
        else:
            adjusted.append(int(center[axis]))
    return np.asarray(adjusted, dtype=int)


def _choose_component(labels, components, rng, used_ids):
    if not components:
        return None, None
    available = [key for key in components if key not in used_ids]
    if not available:
        used_ids.clear()
        available = list(components)
    volumes = np.asarray([components[key]["volume_mm3"] for key in available], dtype=np.float64)
    median_volume = max(float(np.median(volumes)), 1e-6)
    weights = np.clip(np.sqrt(median_volume / np.maximum(volumes, 1e-6)), 0.5, 2.0)
    component_id = rng.choices(available, weights=weights.tolist(), k=1)[0]
    used_ids.add(component_id)
    bounds = components[component_id]["bbox"]
    region = tuple(slice(low, high) for low, high in bounds)
    local_coordinates = np.argwhere(labels[region] == component_id)
    if not len(local_coordinates):
        raise RuntimeError(f"No voxels found for connected component {component_id}")
    origin = np.asarray([low for low, _ in bounds], dtype=int)
    center = local_coordinates[rng.randrange(len(local_coordinates))] + origin
    return component_id, center


def _negative_center(target, roi, rng, support=None, bounds=None, attempts=32):
    """Try to find a label-free crop inside MRI support for explicit negatives."""
    shape = target.shape
    for _ in range(attempts):
        if bounds is not None:
            center = np.asarray([
                rng.randrange(low, max(low + 1, high)) for low, high in bounds
            ], dtype=int)
        else:
            center = np.asarray([rng.randrange(size) for size in shape], dtype=int)
        if support is not None and not support[tuple(center)]:
            continue
        patch = _crop_pad(target, center, (roi,) * 3)
        if not np.any(patch):
            return center
    return np.asarray([rng.randrange(size) for size in shape], dtype=int)


def sample_training_patch(
    images,
    target,
    roi,
    rng,
    *,
    sampling="voxel",
    labels=None,
    components=None,
    used_component_ids=None,
    support=None,
    support_bounds=None,
    negative_probability=0.25,
    local_scale=1.0,
    local_probability=0.0,
):
    """Build aligned image/target/component/support crops for one optimizer update."""
    target = np.asarray(target, dtype=np.uint8)
    if labels is None or components is None:
        labels, components = component_index(target, np.eye(4))
    used_component_ids = used_component_ids if used_component_ids is not None else set()
    selected_id = None

    if sampling == "lesion_balanced" and components and rng.random() >= negative_probability:
        selected_id, center = _choose_component(labels, components, rng, used_component_ids)
    elif sampling == "lesion_balanced":
        center = _negative_center(target, roi, rng, support=support, bounds=support_bounds)
    elif sampling == "voxel":
        positive = np.argwhere(target > 0)
        if len(positive) and rng.random() < 0.7:
            center = positive[rng.randrange(len(positive))]
        else:
            center = np.asarray([rng.randrange(size) for size in target.shape])
        if components:
            selected_id = None
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling}")

    scale = 1.0
    if selected_id is not None and local_probability > 0 and rng.random() < local_probability:
        scale = local_scale
    crop_shape = [max(8, int(round(roi * scale))) for _ in range(3)]
    if selected_id is not None:
        # Retain the selected lesion and a small margin. Large targets use a
        # correspondingly larger source crop before resizing to the fixed ROI.
        for axis, (low, high) in enumerate(components[selected_id]["bbox"]):
            crop_shape[axis] = max(crop_shape[axis], high - low + 8)
        center = _center_for_bbox(components[selected_id], center, crop_shape)

    image_crop = _crop_pad(images, center, crop_shape)
    target_crop = _crop_pad(target, center, crop_shape)
    component_crop = _crop_pad(labels, center, crop_shape)
    support_crop = (
        _crop_pad(support.astype(np.uint8), center, crop_shape).astype(bool)
        if support is not None else np.ones(crop_shape, dtype=bool)
    )
    if crop_shape != [roi] * 3:
        image_crop = _resize_exact(image_crop, roi, order=1)
        target_crop = _resize_exact(target_crop, roi, order=0)
        component_crop = _resize_exact(component_crop, roi, order=0)
        support_crop = _resize_exact(support_crop.astype(np.uint8), roi, order=0).astype(bool)

    return (
        np.ascontiguousarray(image_crop, dtype=np.float32),
        np.ascontiguousarray(target_crop, dtype=np.uint8),
        np.ascontiguousarray(component_crop, dtype=np.int32),
        np.ascontiguousarray(support_crop, dtype=bool),
        selected_id,
    )
