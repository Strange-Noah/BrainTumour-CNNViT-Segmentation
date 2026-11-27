"""Shared model and preprocessing used by Swin UNETR training and inference."""
import numpy as np
from monai.networks.nets import SwinUNETR

from experiments.swin_sampling import cached_acquisition_support


def normalize_modalities(images):
    """Per-volume z-score each modality over nonzero voxels, preserving zero background."""
    result = np.asarray(images, dtype=np.float32).copy()
    if result.ndim != 4 or result.shape[0] != 4:
        raise ValueError(f"Expected 4 x X x Y x Z input, got {result.shape}")
    for channel in result:
        foreground = channel != 0
        if foreground.any():
            values = channel[foreground]
            channel[foreground] = (values - values.mean()) / max(float(values.std()), 1e-6)
            channel[~foreground] = 0
    return np.ascontiguousarray(result)


def build_model(feature_size=24, use_checkpoint=False):
    return SwinUNETR(
        in_channels=4,
        out_channels=2,
        feature_size=int(feature_size),
        spatial_dims=3,
        use_checkpoint=bool(use_checkpoint),
    )
