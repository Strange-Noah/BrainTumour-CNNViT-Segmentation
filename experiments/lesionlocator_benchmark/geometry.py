"""Read-only geometric diagnostics using the official preprocessor and exporter."""
import numpy as np
import torch
from torch.nn import functional as F

from lesionlocator.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from lesionlocator.modules.tracknet import TrackNet


def native_labels(work, properties, plans, configuration):
    native_spacing = [properties["spacing"][i] for i in plans.transpose_forward]
    shape = properties["shape_after_cropping_and_before_resampling"]
    restored = configuration.resampling_fn_seg(work, shape, configuration.spacing,
                                              native_spacing)[0]
    parent = np.zeros(properties["shape_before_cropping"], dtype=restored.dtype)
    parent[tuple(slice(lo, hi) for lo, hi in properties["bbox_used_for_cropping"])] = restored
    return parent.transpose(plans.transpose_backward)


def grid_roundtrip(prompt, shape):
    """TrackNet prompt interpolation independently, without a learned warp."""
    p = torch.from_numpy(np.asarray(prompt, dtype=np.float32))[None]
    grid = F.interpolate(p, size=tuple(shape), mode="nearest-exact")
    restored = F.interpolate(grid, size=p.shape[2:], mode="nearest-exact")
    return grid[0].numpy(), restored[0].numpy()


def geometry_pair(preprocessor, plans, config, dataset, baseline, followup,
                  baseline_labels, followup_labels, registration_shape):
    a, p, pa = preprocessor.run_case([baseline], baseline_labels, plans, config, dataset)
    b, q, pb = preprocessor.run_case([followup], followup_labels, plans, config, dataset)
    a, b, p, q = (x[:] for x in (a, b, p, q))
    p, q = np.maximum(p, 0), np.maximum(q, 0)
    _, _, w0, w1 = TrackNet.optional_z_translation(
        _RegistrationShape(registration_shape), torch.from_numpy(a)[None], torch.from_numpy(b)[None]
    )
    cropped = p[:, w0[0]:w0[1]]
    fixed, returned = grid_roundtrip(cropped, registration_shape)
    combined = np.zeros_like(p)
    combined[:, w0[0]:w0[1]] = returned
    ideal = (q[0] > 0).astype(np.float32)
    restored_ideal = convert_predicted_logits_to_segmentation_with_correct_shape(
        np.stack([1-ideal, ideal]), plans, config, plans.get_label_manager(dataset), pb)
    return dict(G1_native=native_labels(p, pa, plans, config),
                G2_history=w0, G2_followup=w1, G2_retained=cropped,
                G3_fixed=fixed, G3_returned=returned,
                G4_native=native_labels(combined, pa, plans, config),
                G5_native=restored_ideal,
                G5_label_roundtrip=native_labels(q, pb, plans, config))


class _RegistrationShape:
    def __init__(self, shape):
        self.input_shape = tuple(shape)
