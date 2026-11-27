"""Inference-only adapter for the pinned upstream LesionLocator."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import psutil
import torch

from lesionlocator.inference.lesionlocator_segment_and_track import LesionLocatorSegTracker
from lesionlocator.modules import tracknet


def describe_tensor(t):
    return {"shape": list(t.shape), "nonzero": int(torch.count_nonzero(t)),
            "finite": bool(torch.isfinite(t).all())}


def run_case(predictor, baseline, followup, prompt, output, stable_id):
    """One original model step. Only the historical prompt enters this API."""
    h, f, p = map(nib.load, (baseline, followup, prompt))
    if h.shape != p.shape or not np.allclose(h.affine, p.affine, atol=1e-4, rtol=0):
        raise ValueError("History image and prompt grids differ")
    if len(f.shape) != 3 or not np.isfinite(f.affine).all():
        raise ValueError("Invalid follow-up grid")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    log = {"stable_id": stable_id, "T0": {"shape": list(p.shape),
           "prompt_voxels": int(np.count_nonzero(p.dataobj))}}
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    rss = [psutil.Process().memory_info().rss]
    preprocessor = predictor.configuration_manager.preprocessor_class
    orig_case = preprocessor.run_case
    network = predictor.network
    original_z = network.optional_z_translation
    original_roi = network.get_patch_around_mask
    original_warp = tracknet.compute_warped_image_multiNC

    def observe_case(self, *args, **kwargs):
        t = time.monotonic()
        x, y, props = orig_case(self, *args, **kwargs)
        log.setdefault("T1", []).append({"shape": list(x.shape),
            "prompt_voxels": int(np.count_nonzero(y[:] > 0)) if y is not None else None,
            "properties": {k: props[k] for k in ("shape_before_cropping","bbox_used_for_cropping",
                "shape_after_cropping_and_before_resampling","spacing") if k in props},
            "seconds": time.monotonic()-t})
        rss.append(psutil.Process().memory_info().rss)
        return x, y, props

    def observe_z(x0, x1):
        a, b, w0, w1 = original_z(x0, x1)
        log["T2"] = {"history": list(w0), "followup": list(w1),
                     "before": [list(x0.shape), list(x1.shape)],
                     "after": [list(a.shape), list(b.shape)]}
        return a, b, w0, w1

    def observe_warp(prompt_grid, field, spacing, **kwargs):
        log["T3"] = describe_tensor(prompt_grid)
        log["T4"] = {"shape": list(field.shape), "finite": bool(torch.isfinite(field).all()),
                     "coordinate_abs_max": float(field.abs().max()),
                     "coordinate_convention": "official normalized deformation field"}
        warped = original_warp(prompt_grid, field, spacing, **kwargs)
        log["T5_grid"] = describe_tensor(warped)
        return warped

    def observe_roi(warped, is_inference=False):
        slicers = original_roi(warped, is_inference)
        log["T5_work"] = describe_tensor(warped)
        np.save(out / "warped-work.npy", warped.detach().to("cpu").numpy())
        log["T6"] = {"roi_slices": [[int(x) if not isinstance(x, slice) else [x.start, x.stop, x.step]
                                     for x in row] for row in slicers]}
        return slicers

    def on_logits(_, args, output):
        log["T7"] = describe_tensor(output)

    with ExitStack() as stack:
        stack.enter_context(patch.object(preprocessor, "run_case", observe_case))
        stack.enter_context(patch.object(network, "optional_z_translation", observe_z))
        stack.enter_context(patch.object(tracknet, "compute_warped_image_multiNC", observe_warp))
        stack.enter_context(patch.object(network, "get_patch_around_mask", observe_roi))
        stack.callback(network.register_forward_hook(on_logits).remove)
        predictor.track(str(baseline), [str(followup)], str(out), str(prompt))
    paths = list(out.glob("*.nii.gz"))
    if len(paths) > 1:
        raise ValueError("Single-instance call returned multiple predictions")
    if paths:
        prediction = nib.load(paths[0])
        if f.shape != prediction.shape or not np.allclose(f.affine, prediction.affine, atol=1e-4, rtol=0):
            raise ValueError("Official output does not match follow-up native grid")
        values = np.asarray(prediction.dataobj)
        if not np.isin(values, (0, 1)).all():
            raise ValueError("Expected native binary prediction")
        log["T8"] = {"status": "nonempty" if values.any() else "empty",
                     "prediction": paths[0].name, "voxels": int(np.count_nonzero(values))}
    else:
        log["T8"] = {"status": "official_skip"}
    log["resources"] = {"seconds": time.monotonic()-start,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "observed_rss_bytes": max([*rss, psutil.Process().memory_info().rss])}
    (out/"trace.json").write_text(json.dumps(log, default=lambda x: x.tolist()))
    return paths[0] if paths else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "baseline", "followup", "prompt", "output", "stable-id"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")
    predictor = LesionLocatorSegTracker(args.weights, (0,), torch.device("cuda"))
    predictor.use_mirroring = False
    run_case(predictor, args.baseline, args.followup, args.prompt, args.output, args.stable_id)


if __name__ == "__main__":
    main()
