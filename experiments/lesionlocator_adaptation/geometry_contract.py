"""Coordinate and execution contract for the frozen BMETS adaptation."""
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


def resampled_center(coord,input_shape,output_shape):
    """Continuous output-center -> input-center placement for resize grids."""
    return (np.asarray(coord)+.5)*np.asarray(input_shape)/np.asarray(output_shape)-.5


def work_to_native(work, native_crop_shape, work_shape, starts, transpose_forward):
    reader_transposed=resampled_center(work,native_crop_shape,work_shape)+starts
    reader_order=reader_transposed[np.argsort(transpose_forward)]
    # Verified applicable for the checkpoint's SimpleITK reader (z,y,x).
    return reader_order[::-1]


def native_to_work(xyz,native_crop_shape,work_shape,starts,transpose_forward):
    reader_order=np.asarray(xyz)[::-1]
    crop=reader_order[np.asarray(transpose_forward)]-starts
    return resampled_center(crop,work_shape,native_crop_shape)


def work_to_registration(work,work_window_shape,start_z):
    crop=np.asarray(work)-np.array([start_z,0,0])
    return resampled_center(crop,(175,)*3,work_window_shape)


def registration_to_work(reg,work_window_shape,start_z):
    return resampled_center(reg,work_window_shape,(175,)*3)+np.array([start_z,0,0])


def z_threshold(spacing):
    if not np.isfinite(spacing) or spacing<=0:
        raise ValueError("Invalid depth spacing")
    return int(np.floor(60./spacing+1e-8))


def sha(path):
    with Path(path).open("rb") as f:
        return file_digest(f,"sha256").hexdigest()


def prepare_tracker(weights,freeze,device):
    """Build the next authorized execution from exact private model and plans bytes."""
    from lesionlocator.inference.lesionlocator_segment_and_track import LesionLocatorSegTracker
    from experiments.lesionlocator_benchmark.roi_padding import pad_spatial,spatial_padding,unpad_spatial

    folder=Path(weights)
    if sha(folder/"fold_0/checkpoint_final.pth")!=freeze["model_sha256"]:
        raise ValueError("Official weights differ from the frozen checkpoint")
    if sha(folder/"plans.json")!=freeze["plans_sha256"]:
        raise ValueError("Input and native export must share the frozen plans")
    if sha(folder/"dataset.json")!=freeze["dataset_sha256"]:
        raise ValueError("Official dataset definition changed")
    p=LesionLocatorSegTracker(str(folder),(0,),device)
    p.use_mirroring=False
    if tuple(p.configuration_manager.spacing)!=tuple(freeze["spacing"]):
        raise ValueError("Loaded preprocessor/exporter spacing disagrees with the freeze")
    network=p.network
    if tuple(network.input_shape)!=(175,)*3 or tuple(network.unet_patch_size.tolist())!=(96,256,256):
        raise ValueError("Tracking model tensor dimensions changed")
    original_z=network.optional_z_translation
    original_pad,original_unpad=network.pad_input,network.undo_pad
    threshold=z_threshold(freeze["spacing"][0])
    network.optional_z_translation=lambda a,b:original_z(a,b,difference_threshold=threshold)

    def pad(value):
        widths=spatial_padding(value,network.unet_patch_size)
        if any(widths[2:]):
            return pad_spatial(value,widths),True,("spatial",widths)
        return original_pad(value)

    def unpad(value,padded,widths):
        if padded and widths[0]=="spatial":
            return unpad_spatial(value,widths[1])
        return original_unpad(value,padded,widths)

    network.pad_input=pad
    network.undo_pad=unpad
    return p
