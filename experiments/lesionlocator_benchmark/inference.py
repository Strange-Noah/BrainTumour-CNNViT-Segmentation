"""Fixed-protocol, validation-only LesionLocator GPU and geometry worker."""
import hashlib
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import psutil
import torch
from lesionlocator.inference.lesionlocator_segment_and_track import LesionLocatorSegTracker
from lesionlocator.utilities.plans_handling.plans_handler import PlansManager

from research.evaluate import evaluate
from research.geometry import mask_geometry
from .baseline import instances, json_scalar
from .geometry import normalization, overlap, run_pair, _instance_report
from .tracker import run_case


def write_mask(mask, path, ref):
    header = ref.header.copy()
    header.set_data_dtype(mask.dtype)
    nib.save(nib.Nifti1Image(mask, ref.affine, header), path)


def visit(files, root, index):
    path = files["image"]
    ref, seg = map(nib.load, (path, files["seg"]))
    mask_geometry(seg, ref, files["exam"])
    labels = instances(np.asarray(seg.dataobj))
    instance_file = root/f"instances-{index}.nii.gz"
    write_mask(labels.astype(np.int32), instance_file, ref)
    return dict(image=path, ref=ref, labels=labels, instances=instance_file)


def compare(mask, target, ref, identity, reference_proxy):
    result = {"id": identity, "voxels": int(np.count_nonzero(mask)),
              "reference_proxy_id": reference_proxy, "reference_proxy": None}
    if reference_proxy is not None:
        truth = target == reference_proxy
        if not truth.any():
            raise ValueError("Accepted L0 reference proxy missing from the frozen target")
        result["reference_proxy"] = _instance_report(reference_proxy, truth, mask, ref.affine)
    return result


def load_mask(path, target):
    return np.asarray(nib.load(path).dataobj) > 0 if path else np.zeros(target.shape, dtype=bool)


def evaluate_pass(predictor, history, target, prompt_ids, prompts, accepted, output, mode, smoke=False):
    historical = history["labels"]
    truth = target["labels"]
    proxies = {r["instance_id"]: r for r in accepted["instances"]}
    combined = np.zeros(truth.shape, dtype=np.uint16)
    warps = np.zeros_like(combined)
    instances_report, predicted = [], {}
    pre = predictor.configuration_manager.preprocessor_class(verbose=False)
    plans, conf, data = predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json
    diagnostic = None
    for identity in prompt_ids:
        prompt = prompts.get(identity)
        status = "no_previous_prediction"
        folder = output / f"component-{identity}"
        trace = None
        warp = final = None
        if prompt is not None:
            final = run_case(predictor, history["image"], target["image"], prompt, folder, identity, save_field=smoke and identity==prompt_ids[0])
            trace = json.loads((folder/"trace.json").read_text())
            warp = folder/"warp.nii.gz"
            if not warp.exists():
                warp = None
            status = trace["T8"]["status"]
        if final is not None and status == "nonempty":
            predicted[identity] = final
        final_mask, warped = (load_mask(path, truth) for path in (final, warp))
        proxy = proxies.get(identity, {}) if mode == "L1" else {}
        reference = proxy.get("reference_proxy", {}).get("target_id")
        combined += final_mask
        warps += warped
        row = dict(id=identity, status=status, T9={"source": ("prior_gt" if mode == "L1" else
                   "initial_gt" if accepted.get("initial") else "prior_prediction"),
                   "carried_to_next_step": identity in predicted},
                   L0=proxy or None,
                   warp=compare(warped, truth, target["ref"], identity, reference),
                   final=compare(final_mask, truth, target["ref"], identity, reference))
        if trace and "T6" in trace:
            roi = load_mask(folder/"roi.nii.gz", truth)
            union_gt = truth > 0
            related = truth == reference if reference is not None else None
            z_visible = load_mask(folder/"z-visible.nii.gz", truth)
            row["T6"] = {"z_union_gt_coverage": overlap(union_gt, z_visible)["recall"],
                         "z_reference_coverage": None if related is None else overlap(related, z_visible)["recall"],
                         "union_gt_coverage": overlap(union_gt, roi)["recall"],
                         "reference_coverage": None if related is None else overlap(related, roi)["recall"]}
            # Only the offline observer reads target GT. The model already returned.
            if diagnostic is None:
                _, label, properties = pre.run_case([str(target["image"])], [str(target["instances"])],
                                                      plans, conf, data)
                diagnostic = (label, properties)
            label, props = diagnostic
            official_props = trace["T1"][0]["properties"]
            if any(not np.array_equal(value, props[k]) for k, value in official_props.items()):
                raise ValueError("Offline ROI scoring uses a different physical transform")
            z0 = trace["T2"]["followup"][0]
            a, b, c, d, e, f = trace["T6"]["roi_slices"][0]
            patch = label[0, z0+a:z0+b, c:d, e:f]
            local = np.load(folder/"local-prediction.npy", allow_pickle=False)
            offsets = (np.asarray(local.shape)-patch.shape)//2
            real = local[tuple(slice(o, o+n) for o, n in zip(offsets, patch.shape))]
            row["T7"] = {"union_gt": overlap(patch > 0, real),
                         "reference_gt": None if reference is None else overlap(patch == reference, real)}
            row["T8"] = {"work_vs_final_on_native": overlap(
                load_mask(folder/"work-logits-argmax.nii.gz", truth), final_mask)}
        instances_report.append(row)
    affine = target["ref"].affine
    return dict(instances=instances_report,
                union=evaluate(combined > 0, truth > 0, affine),
                warp_union=evaluate(warps > 0, truth > 0, affine),
                overlapping_final_voxels=int(np.count_nonzero(combined > 1))), predicted


def process_patient(model, active, files, l0, root, device, geometry_only, smoke=False):
    if geometry_only:
        plans = PlansManager(json.loads((model/"plans.json").read_text()))
        data = json.loads((model/"dataset.json").read_text())
        conf = plans.get_configuration(active["configuration"])
        predictor = None
    else:
        predictor = model
        plans, conf, data = predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json
    results, nstats = [], []
    carry = {}
    initial_ids = []
    previous = None
    for index, file in enumerate(files[:2] if smoke else files):
        current = visit(file, root, index)
        image_plot = root/f"intensity-{index}.png" if index < 2 else None
        nstats.append(normalization(current, active, image_plot))
        if previous is None:
            previous = current
            initial_ids = [int(i) for i in np.unique(current["labels"]) if i]
            continue
        pair_dir = root/f"pair-{index}"
        pair_dir.mkdir()
        accepted = l0["direct_l0"][index-1]
        if (accepted["historical"], accepted["followup"]) != (files[index-1]["exam"], file["exam"]):
            raise ValueError("Accepted baseline belongs to a different validation pair")
        if geometry_only or smoke:
            gp = pair_dir/"geometry"
            gp.mkdir()
            geometry = run_pair(plans, conf, data, active, previous, current, gp, device)
        else:
            geometry = None
        entry = dict(geometry=geometry, L0=accepted["metrics"])
        if predictor:
            historical = previous["labels"]
            ids = [int(i) for i in np.unique(historical) if i]
            prompts = {}
            for identity in ids:
                path = pair_dir/f"history-{identity}.nii.gz"
                write_mask((historical == identity).astype(np.uint8), path, previous["ref"])
                prompts[identity] = path
            entry["L1"], first_predictions = evaluate_pass(
                predictor, previous, current, ids, prompts, accepted, pair_dir/"L1", "L1", smoke)
            if index == 1:
                entry["L2"] = {"shared_initial_l1_output": True, "initial_ids": initial_ids,
                               "missing_after_step": sorted(set(initial_ids)-set(first_predictions))}
                carry = first_predictions
            else:
                entry["L2"], carry = evaluate_pass(
                    predictor, previous, current, initial_ids, carry, dict(instances=[]),
                    pair_dir/"L2", "L2")
        results.append(entry)
        previous = current
    return dict(pairs=results, normalization=nstats, exams=len(results)+1)


def predictor(model, active, device):
    p = LesionLocatorSegTracker(str(model), (0,), device)
    p.use_mirroring = False
    if list(p.network.input_shape) != active["registration_shape"] or p.trainer_name != active["trainer"]:
        raise ValueError("Loaded model does not match official inspected configuration")
    return p
