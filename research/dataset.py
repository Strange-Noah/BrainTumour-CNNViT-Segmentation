"""Cached R2 loader. Returns original-grid T1,T1c,T2,FLAIR and raw labels."""
import csv
import hashlib
import json
from pathlib import Path
import os
import tempfile

import nibabel as nib
import numpy as np

from cloud.storage import DATASET, connect
from research.evaluate import target
from research.geometry import mask_geometry, require_mm


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    """Hash a file in bounded memory rather than reading the whole scan at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Dataset:
    def __init__(self, manifest, split, cache, partition="train"):
        split_bytes = Path(split).read_bytes()
        if hashlib.sha256(split_bytes).hexdigest() != "57f5af55141e16ecb0b85fdda63cc355175d9b35271126784eb22766698b36f0":
            raise ValueError("Use the frozen Protocol v1 patient split")
        manifest_bytes = Path(manifest).read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != "040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4":
            raise ValueError("Use the canonical Protocol v1 manifest")
        if partition not in ("train", "validation", "test"):
            raise ValueError("Unknown split partition")
        mapping = list(csv.DictReader(split_bytes.decode().splitlines()))
        patients = [row["patient_id"] for row in mapping]
        if len(patients) != len(set(patients)):
            raise ValueError("Duplicate split patients")
        selected = {row["patient_id"] for row in mapping if row["partition"] == partition}
        self.visits = [json.loads(line) for line in Path(manifest).read_text().splitlines()
                       if line.strip()]
        self.visits = [row for row in self.visits if row["patient_id"] in selected]
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.client, self.bucket = connect()
        # A visit is loaded repeatedly across epochs. Remember successful cache
        # verification so we avoid an R2 HEAD request and full-file hash each time.
        # The file's size and mtime are checked on every hit; a changed file is
        # re-fetched and checksum-verified.
        self._verified_objects = {}

    def fetch(self, relative):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Invalid raw path")
        verified = self._verified_objects.get(relative)
        if verified is not None:
            path, size, mtime_ns = verified
            try:
                stat = path.stat()
                if stat.st_size == size and stat.st_mtime_ns == mtime_ns:
                    return path
            except FileNotFoundError:
                pass
            self._verified_objects.pop(relative, None)

        key = f"{DATASET}/raw/{relative}"
        head = self.client.head_object(Bucket=self.bucket, Key=key)
        digest = head.get("Metadata", {}).get("sha256")
        if not digest:
            raise ValueError("Missing ingestion checksum")
        path = self.cache / (digest + (".nii.gz" if key.endswith(".gz") else ".nii"))
        if path.exists() and _sha256_file(path) == digest:
            stat = path.stat()
            self._verified_objects[relative] = (path, stat.st_size, stat.st_mtime_ns)
            return path
        fd, name = tempfile.mkstemp(dir=self.cache)
        try:
            with os.fdopen(fd, "wb") as output:
                self.client.download_fileobj(self.bucket, key, output)
            if _sha256_file(name) != digest:
                raise ValueError("Data checksum mismatch")
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
        stat = path.stat()
        self._verified_objects[relative] = (path, stat.st_size, stat.st_mtime_ns)
        return path

    def load(self, index, task="enhancing"):
        row = self.visits[index]
        paths = {name: self.fetch(path) for name, path in row["modalities"].items()}
        reference = nib.load(paths["t1c"])
        require_mm(reference, "T1c reference")
        warnings = []
        arrays = []
        # Explicit resampling to T1c accommodates differing native modality grids.
        from nibabel.processing import resample_from_to
        for name in ("t1n", "t1c", "t2", "t2f"):
            image = nib.load(paths[name])
            require_mm(image, name)
            if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
                image = resample_from_to(image, reference, order=1)
            arrays.append(image.get_fdata(dtype=np.float32))
        label = nib.load(self.fetch(row["segmentation"]))
        warnings.extend(mask_geometry(label, reference, row["exam_id"]))
        raw = label.get_fdata()
        if not np.isfinite(raw).all() or not np.isin(raw, [0, 1, 2, 3, 4]).all():
            raise ValueError("Unexpected source labels")
        return {"images": np.stack(arrays), "raw_labels": raw.astype(np.uint8),
                "target": target(raw, task), "reference": reference, "visit": row, "geometry_warnings": warnings}
