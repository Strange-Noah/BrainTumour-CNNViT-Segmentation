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

from lesionlocator.inference import lesionlocator_segment_and_track as upstream
from lesionlocator.inference.lesionlocator_segment_and_track import LesionLocatorSegTracker
from lesionlocator.modules import tracknet
from .geometry import native_labels
from .roi_padding import pad_spatial, spatial_padding, unpad_spatial


def describe_tensor(t):
    return {"shape": list(t.shape), "nonzero": int(torch.count_nonzero(t)),
            "finite": bool(torch.isfinite(t).all())}


def run_case(predictor, baseline, followup, prompt, output, stable_id, save_field=False):
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
    context = {}
    original_z = network.optional_z_translation
    original_roi = network.get_patch_around_mask
    original_pad, original_unpad = network.pad_input, network.undo_pad
    original_warp = tracknet.compute_warped_image_multiNC
    original_export = upstream.export_prediction_from_logits
    events = {}

    def start_event(name):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        events[name] = start, end
        start.record()

    def measure_export(*args, **kwargs):
        before = time.monotonic()
        result = original_export(*args, **kwargs)
        log["native_export_seconds"] = time.monotonic()-before
        return result

    def observe_case(self, *args, **kwargs):
        t = time.monotonic()
        x, y, props = orig_case(self, *args, **kwargs)
        if str(args[0][0]) == str(followup):
            if args[1] is not None:
                raise AssertionError("Target annotation entered inference")
            context["target_properties"] = props
            context["target_shape"] = x.shape
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
                     "dtype": str(x0.dtype), "before": [list(x0.shape), list(x1.shape)],
                     "after": [list(a.shape), list(b.shape)]}
        return a, b, w0, w1

    def observe_warp(prompt_grid, field, spacing, **kwargs):
        log["T3"] = describe_tensor(prompt_grid)
        log["T4"] = {"shape": list(field.shape), "finite": bool(torch.isfinite(field).all()),
                     "coordinate_abs_max": float(field.abs().max()),
                     "coordinate_convention": "official normalized deformation field"}
        field_sample = field[..., ::4, ::4, ::4].detach().float().cpu().numpy()
        grid_sample = network.reg_net.identity_map[..., ::4, ::4, ::4].detach().float().cpu().numpy()
        if field_sample.shape == grid_sample.shape:
            norm = np.linalg.norm(field_sample-grid_sample, axis=1)
            log["T4"]["normalized_displacement_quantiles"] = np.quantile(norm, [.5,.95,1]).tolist()
        if save_field:
            np.save(out/"deformation.npy", field.detach().cpu().numpy(), allow_pickle=False)
        warped = original_warp(prompt_grid, field, spacing, **kwargs)
        log["T5_grid"] = describe_tensor(warped)
        return warped

    def observe_roi(warped, is_inference=False):
        slicers = original_roi(warped, is_inference)
        log["T5_work"] = describe_tensor(warped)
        context["warp"] = warped[0].detach().to("cpu").numpy()
        log["T6"] = {"roi_slices": [[int(x) if not isinstance(x, slice) else [x.start, x.stop, x.step]
                                     for x in row] for row in slicers]}
        return slicers

    def apply_pad(tensor):
        widths = spatial_padding(tensor, network.unet_patch_size)
        fix = any(widths[2:])
        log["T6"].update(work_roi=list(tensor.shape[-3:]), spatial_padding=list(widths),
                         axis_compatibility_fix=fix,
                         warp_empty=not bool(log["T5_work"]["nonzero"]))
        context["spatial_padding_fix"] = fix
        if fix:
            # Upstream places the H/W pairs in the C/N slots of the F.pad list.
            return pad_spatial(tensor, widths), True, widths
        return original_pad(tensor)

    def undo_pad(tensor, padded, widths):
        return unpad_spatial(tensor, widths) if context["spatial_padding_fix"] else original_unpad(tensor, padded, widths)

    def on_roi_logits(_, args, output):
        events["unet"][1].record()
        log["T7"] = dict(describe_tensor(output), input_shape=list(args[0].shape))
        if not log["T7"]["finite"]:
            raise FloatingPointError("Local U-Net logits are not finite")
        context["local_prediction"] = output[0].argmax(0).detach().cpu().numpy().astype(np.uint8)

    def on_full_logits(_, args, output):
        log["T8_work"] = describe_tensor(output)
        if not log["T8_work"]["finite"]:
            raise FloatingPointError("TrackNet logits are not finite")
        context["work_prediction"] = output[0].argmax(0).detach().cpu().numpy().astype(np.uint8)

    with ExitStack() as stack:
        stack.enter_context(patch.object(upstream, "export_prediction_from_logits", measure_export))
        stack.enter_context(patch.object(preprocessor, "run_case", observe_case))
        stack.enter_context(patch.object(network, "optional_z_translation", observe_z))
        stack.enter_context(patch.object(tracknet, "compute_warped_image_multiNC", observe_warp))
        stack.enter_context(patch.object(network, "get_patch_around_mask", observe_roi))
        stack.enter_context(patch.object(network, "pad_input", apply_pad))
        stack.enter_context(patch.object(network, "undo_pad", undo_pad))
        stack.callback(network.reg_net.register_forward_pre_hook(
            lambda *_: start_event("registration")).remove)
        def end_registration(*_):
            events["registration"][1].record()
        stack.callback(network.reg_net.register_forward_hook(end_registration).remove)
        stack.callback(network.unet.register_forward_pre_hook(lambda *_: start_event("unet")).remove)
        stack.callback(network.unet.register_forward_hook(on_roi_logits).remove)
        stack.callback(network.register_forward_hook(on_full_logits).remove)
        predictor.track(str(baseline), [str(followup)], str(out), str(prompt))
    if events:
        torch.cuda.synchronize()
        log["cuda_seconds"] = {name: first.elapsed_time(last)/1000
                               for name, (first,last) in events.items()}
    paths = list(out.glob("*.nii.gz"))
    if "warp" in context:
        props = context["target_properties"]
        plans = predictor.plans_manager
        config = predictor.configuration_manager
        z0, z1 = log["T2"]["followup"]
        full = np.zeros(context["target_shape"], dtype=np.uint8)
        full[:, z0:z1] = context["warp"] > 0
        roi = np.zeros_like(full)
        a, b, c, d, e, g = log["T6"]["roi_slices"][0]
        roi[:, z0+a:z0+b, c:d, e:g] = 1

        def write_native(work, name):
            array = native_labels(work, props, plans, config)
            path = out / (name+".nii.gz")
            plans.image_reader_writer_class().write_seg(array, str(path), props)
            result = nib.load(path)
            if result.shape != f.shape or not np.allclose(result.affine, f.affine, rtol=0, atol=1e-4):
                raise ValueError("Diagnostic output does not match follow-up grid")
            return int(np.count_nonzero(result.dataobj))

        log["T5_native_voxels"] = write_native(full, "warp")
        log["T6"]["native_roi_voxels"] = write_native(roi, "roi")
        z_visible = np.zeros_like(full)
        z_visible[:, z0:z1] = 1
        log["T6"]["z_visible_native_voxels"] = write_native(z_visible, "z-visible")
        log["T8_work"]["foreground_voxels"] = int(context["work_prediction"].sum())
        log["T8_work"]["nearest_native_voxels"] = write_native(context["work_prediction"][None], "work-logits-argmax")
        np.save(out/"local-prediction.npy", context["local_prediction"], allow_pickle=False)
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
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
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
