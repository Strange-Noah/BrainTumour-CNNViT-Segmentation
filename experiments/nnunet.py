"""Qiu entry point for a task-specific nnU-Net v2 trained dataset."""
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def predict(sample):
    import nibabel as nib
    import numpy as np
    with TemporaryDirectory() as name:
        root = Path(name)
        source, output = root / "input", root / "output"
        source.mkdir()
        reference = sample["reference"]
        for index, image in enumerate(sample["images"]):
            nib.save(nib.Nifti1Image(image, reference.affine, reference.header),
                     source / f"visit_{index:04d}.nii.gz")
        subprocess.run(["nnUNetv2_predict", "-i", str(source), "-o", str(output),
                        "-d", os.environ["NNUNET_DATASET_ID"], "-c", os.environ.get("NNUNET_CONFIG", "3d_fullres"),
                        "-f", os.environ.get("NNUNET_FOLD", "0")], check=True)
        prediction = nib.load(output / "visit.nii.gz")
        if prediction.shape != reference.shape or not np.allclose(prediction.affine, reference.affine, atol=1e-4):
            raise ValueError("nnU-Net output geometry differs from reference")
        values = prediction.get_fdata()
        if not np.isin(values, [0, 1]).all():
            raise ValueError("Use a task-specific binary checkpoint")
        return values.astype(np.uint8)
