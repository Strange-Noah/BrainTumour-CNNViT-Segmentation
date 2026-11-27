"""Fetch only official tracking assets on the trusted cloud CPU runner."""
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile

from cloud.storage import connect, put_once
from .baseline import save_once

URL = "https://zenodo.org/records/15174217/files/LesionLocatorModel.zip"
SIZE = 14073990099
UPSTREAM = "ef073baaa0dcba8dd036ec5a29f130db18c9d580"
PREFIX = "novo-nordisk/experiments/lesionlocator/assets/zenodo-15174217/fold0"
MEMBERS = ("LesionLocatorTrack/dataset.json", "LesionLocatorTrack/plans.json",
           "LesionLocatorTrack/fold_0/checkpoint_final.pth")


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, SIZE)[whence] + offset
        if self.position < 0:
            raise ValueError("Invalid ZIP offset")
        return self.position

    def read(self, length=-1):
        left = self.position
        length = min(length if length >= 0 else SIZE - left, SIZE - left)
        if length <= 0:
            return b""
        if length > 16 * 1024 * 1024:
            raise ValueError("Unexpected oversized ZIP read")
        right = left + length - 1
        request = Request(URL, headers={"Range": f"bytes={left}-{right}",
                                        "Accept-Encoding": "identity"})
        with urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers["Content-Range"] != f"bytes {left}-{right}/{SIZE}":
                raise ValueError("Official ZIP range mismatch")
            data = response.read(length + 1)
        if len(data) != length:
            raise ValueError("Incomplete official ZIP range")
        self.position += length
        return data


def main():
    writer, bucket = connect(output=True)
    if bucket != "novo-nordisk-pbl-outputs":
        raise ValueError("Private output bucket required")
    with tempfile.TemporaryDirectory(prefix="lesionlocator-assets-") as folder, ZipFile(RemoteZip()) as archive:
        metadata = {}
        for member in MEMBERS:
            candidates = [name for name in archive.namelist()
                          if name == member or name.endswith("/" + member)]
            if len(candidates) != 1:
                raise ValueError(f"Expected one official archive entry for {member}")
            name = candidates[0]
            info = archive.getinfo(name)
            if info.file_size > 4 * 1024**3:
                raise ValueError("Unexpected official member size")
            target = Path(folder) / Path(member).name
            # ZipExtFile validates CRC on reaching EOF.
            with archive.open(name) as source, target.open("xb") as dest:
                shutil.copyfileobj(source, dest, length=8 * 1024**2)
            if target.stat().st_size != info.file_size:
                raise ValueError("Official member byte size mismatch")
            result = put_once(writer, bucket, f"{PREFIX}/{member}", target)
            metadata[member] = dict(result, archive_member=name, zip_crc32=f"{info.CRC:08x}")
            if target.suffix == ".json":
                document = json.loads(target.read_text())
                if target.name == "dataset.json":
                    metadata[member]["config"] = {k: document.get(k) for k in ("channel_names","labels","file_ending")}
                else:
                    metadata[member]["config"] = {k: document.get(k) for k in ("transpose_forward","transpose_backward","configurations")}
        save_once(writer,bucket,f"{PREFIX}/receipt.json",
                  dict(record=15174217,upstream=UPSTREAM,archive_size=SIZE,
                       verification="range extraction with per-member ZIP CRC32 and R2 SHA-256; full ZIP MD5 not computed",
                       members=metadata))
        print(json.dumps(dict(status="tracking_assets_verified", members=metadata), allow_nan=False))


if __name__ == "__main__":
    main()
