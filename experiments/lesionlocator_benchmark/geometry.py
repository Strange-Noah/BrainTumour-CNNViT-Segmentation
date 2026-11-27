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


def main():
    """Pilot G1–G5 on the first frozen validation pair, CPU only."""
    import json
    from pathlib import Path
    import tempfile

    import nibabel as nib
    from lesionlocator.utilities.plans_handling.plans_handler import PlansManager
    from cloud.storage import connect
    from research.dataset import Dataset
    from .assets import PREFIX as ASSETS
    from .baseline import MANIFEST_KEY, SPLIT, load_visit, save_once, sequences

    torch.set_num_threads(2)
    reader, bucket = connect()
    writer, output = connect(output=True)
    if bucket != "novo-nordisk-pbl" or output != "novo-nordisk-pbl-outputs":
        raise ValueError("Expected private project storage")

    def read_json(key):
        with writer.get_object(Bucket=output, Key=key)["Body"] as body:
            return json.load(body)

    def metric(a, b):
        a, b = a > 0, b > 0
        if a.shape != b.shape:
            raise ValueError("Unequal grid shapes")
        total = np.count_nonzero(a)+np.count_nonzero(b)
        return dict(voxels_in=int(a.sum()), voxels_out=int(b.sum()),
                    dice=float(2*(a & b).sum()/total) if total else None)

    with tempfile.TemporaryDirectory(prefix="novo-ll-geometry-") as d:
        root = Path(d)
        accepted = read_json(ASSETS+"/receipt.json")
        from cloud.storage import verify
        def config_asset(name):
            r = accepted["members"]["LesionLocatorTrack/"+name]
            with (root/name).open("wb") as f:
                verify(writer, output, r["key"], r["sha256"], f)
            return json.loads((root/name).read_text())
        plans = PlansManager(config_asset("plans.json"))
        data = config_asset("dataset.json")
        # The active configuration must be extracted from the checkpoint
        # metadata by the isolated official runtime before any medical run.
        active = read_json(ASSETS+"/runtime.json")
        ckpt = accepted["members"]["LesionLocatorTrack/fold_0/checkpoint_final.pth"]["sha256"]
        if active["checkpoint_sha256"] != ckpt:
            raise ValueError("Checkpoint runtime identity mismatch")
        config = plans.get_configuration(active["configuration"])
        path = root/"visits.jsonl"
        reader.download_file(bucket, MANIFEST_KEY, str(path))
        ds = Dataset(path, SPLIT, root/"images", partition="validation")
        first = next(iter(sequences(ds.visits).values()))[:2]
        args, gt = [], []
        for i,row in enumerate(first):
            ref, labels, _, _ = load_visit(ds,row)
            source = ds.fetch(row["modalities"]["t1c"])
            label = root/f"instances-{i}.nii.gz"
            nib.save(nib.Nifti1Image(labels.astype(np.int32),ref.affine,ref.header), label)
            args.append((str(source),str(label)))
            gt.append(labels)
        results = geometry_pair(config.preprocessor_class(verbose=False), plans, config,
                                data, args[0][0], args[1][0], args[0][1], args[1][1],
                                active["registration_shape"])
        stats = {"G1":metric(gt[0],results["G1_native"]),
                 "G4":metric(gt[0],results["G4_native"]),
                 "G5":metric(gt[1],results["G5_native"]),
                 "G5_label_reference":metric(gt[1],results["G5_label_roundtrip"]),
                 "G2_windows":[results["G2_history"],results["G2_followup"]],
                 "G3_grid_shape":list(results["G3_fixed"].shape)}
        save_once(writer,output,ASSETS+"/geometry-pilot-"+ckpt+".json",stats)
        print(json.dumps(dict(status="geometry_pilot", **stats)))


if __name__ == "__main__":
    main()
