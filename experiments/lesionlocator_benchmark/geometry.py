"""Official inverse geometry diagnostics with native-coordinate instance scoring."""
import numpy as np
import nibabel as nib
import torch
from torch.nn import functional as F
from lesionlocator.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from lesionlocator.modules.tracknet import TrackNet
from .baseline import describe
from .normalization import characterize


def native_labels(work, properties, plans, configuration):
    native_spacing = [properties["spacing"][i] for i in plans.transpose_forward]
    shape = properties["shape_after_cropping_and_before_resampling"]
    restored = np.asarray(configuration.resampling_fn_seg(
        work, shape, configuration.spacing, native_spacing))[0]
    parent = np.zeros(properties["shape_before_cropping"], dtype=restored.dtype)
    parent[tuple(slice(lo, hi) for lo, hi in properties["bbox_used_for_cropping"])] = restored
    return parent.transpose(plans.transpose_backward)


def native_array(array, plans, props, path, ref):
    plans.image_reader_writer_class().write_seg(array, str(path), props)
    image = nib.load(path)
    if image.shape != ref.shape or not np.allclose(image.affine, ref.affine, atol=1e-4, rtol=0):
        raise ValueError("Official inverse mapping disagrees with native T1c")
    return np.asarray(image.dataobj)


def overlap(a, b):
    a, b = a > 0, b > 0
    n, m = int(a.sum()), int(b.sum())
    common = int(np.count_nonzero(a & b))
    return dict(reference_voxels=n, predicted_voxels=m, intersect=common,
                dice=2*common/(n+m) if n+m else None, recall=common/n if n else None)


def _instance_report(identity, ref_mask, pred_mask, affine):
    history = describe(ref_mask.astype(np.uint8), affine)[0]
    found = describe(pred_mask.astype(np.uint8), affine)
    found = found[0] if found else dict(voxels=0, volume_mm3=0., centroid_mm=None)
    return dict(id=identity, exists=bool(found["voxels"]), native=found,
                volume_change_mm3=found["volume_mm3"]-history["volume_mm3"],
                centroid_delta_mm=(np.subtract(found["centroid_mm"], history["centroid_mm"]).tolist()
                                   if found["voxels"] else None),
                **overlap(ref_mask, pred_mask))


class _RegistrationShape:
    def __init__(self, shape):
        self.input_shape = tuple(shape)


def run_pair(plans, config, dataset, active, history, target, root, device):
    pre = config.preprocessor_class(verbose=False)
    h, p, hp = pre.run_case([str(history["image"])], [str(history["instances"])], plans, config, dataset)
    f, _, fp = pre.run_case([str(target["image"])], None, plans, config, dataset)
    # Diagnostic-only read after the unlabelled follow-up has been preprocessed.
    diagnostic, q, qp = pre.run_case([str(target["image"])], [str(target["instances"])], plans, config, dataset)
    if f.shape != diagnostic.shape or any(not np.array_equal(fp[k], qp[k]) for k in (
        "spacing", "bbox_used_for_cropping", "shape_before_cropping",
        "shape_after_cropping_and_before_resampling")):
        raise ValueError("Diagnostic and model follow-up transforms differ")
    p, q = np.maximum(p, 0), np.maximum(q, 0)
    x0, x1 = [torch.from_numpy(a.astype(np.float16))[None].to(device) for a in (h, f)]
    shape = active["registration_shape"]
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        _, _, wh, wf = TrackNet.optional_z_translation(_RegistrationShape(shape), x0, x1)
    prompt = torch.from_numpy(p[:, wh[0]:wh[1]].astype(np.float16))[None].to(device)
    fixed = F.interpolate(prompt, size=shape, mode="nearest-exact")
    back = F.interpolate(fixed, size=prompt.shape[2:], mode="nearest-exact")[0].cpu().numpy().astype(p.dtype)
    crop, returned = np.zeros_like(p), np.zeros_like(p)
    crop[:, wh[0]:wh[1]] = p[:, wh[0]:wh[1]]
    returned[:, wh[0]:wh[1]] = back
    ref = history["labels"]
    stages = {}
    for stage, work in (("G1", p), ("G2", crop), ("G4", returned)):
        a = native_labels(work, hp, plans, config)
        n = native_array(a, plans, hp, root/(stage+".nii.gz"), history["ref"])
        stages[stage] = dict(union=overlap(ref, n), instances=[
            _instance_report(i, ref == i, n == i, history["ref"].affine)
            for i in np.unique(ref) if i])
    unique, counts = np.unique(fixed[0, 0].cpu().numpy(), return_counts=True)
    grid_counts = dict(zip(map(int, unique), map(int, counts)))
    stages["G3"] = dict(shape=list(shape), history_window=wh, followup_window=wf,
                         convention="nearest-exact centers on z-cropped work arrays; no fixed physical spacing",
                         source_work_shape=list(h.shape[1:]), target_work_shape=list(f.shape[1:]),
                         instances=[dict(id=int(i), grid_voxels=grid_counts.get(i, 0),
                                         exists=i in grid_counts) for i in np.unique(ref) if i])
    stages["G2"].update(windows=[wh, wf], input_dtype=str(x0.dtype), backend=str(device),
                        cuda_autocast=device.type == "cuda")
    fref = target["labels"]
    model_labels = plans.get_label_manager(dataset)

    def export(mask, name):
        fg = mask.astype(np.float32)
        recovered = convert_predicted_logits_to_segmentation_with_correct_shape(
            np.stack([1-fg, fg]), plans, config, model_labels, fp)
        return native_array(recovered, plans, fp, root/name, target["ref"])

    combined = export(q[0] > 0, "G5.nii.gz")
    per_lesion = []
    for i in np.unique(fref):
        if i:
            recovered = export(q[0] == i, f"G5-{i}.nii.gz")
            per_lesion.append(_instance_report(i, fref == i, recovered > 0, target["ref"].affine))
    stages["G5"] = dict(union=overlap(fref, combined), instances=per_lesion)
    return stages


def normalization(visit, active, plot=None):
    if active["configuration_fields"]["normalization_schemes"] != ["CTNormalization"]:
        return {"scheme": active["configuration_fields"]["normalization_schemes"]}
    parameters = active["foreground_intensity_properties_per_channel"]["0"]
    image = visit["ref"].get_fdata(dtype=np.float32, caching="unchanged")
    target = visit["labels"] > 0
    if plot is not None and target.any():
        import matplotlib.pyplot as plt
        k = int(target.sum(axis=(0, 1)).argmax())
        sl = image[:, :, k]
        norm = (np.clip(sl, parameters["percentile_00_5"], parameters["percentile_99_5"])
                - parameters["mean"])/parameters["std"]
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        for ax, data, title in zip(axs, (sl, norm), ("Native MRI", "Fixed CT transform")):
            ax.imshow(data.T, cmap="gray", origin="lower")
            ax.contour(target[:, :, k].T, levels=[.5], colors="yellow", linewidths=.7, origin="lower")
            ax.set_title(title)
            ax.axis("off")
        fig.savefig(plot, dpi=120)
        plt.close(fig)
    return characterize(image, target, parameters)
