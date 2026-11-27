"""Trusted control plane for private LesionLocator inference and independent readback."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
from zipfile import ZipFile

from botocore.exceptions import ClientError
from kaggle.api.kaggle_api_extended import KaggleApi
from cloud.storage import DATASET, connect
from research.dataset import Dataset
from .aggregate import Diagnostics
from .assets import MEMBERS, PREFIX as ASSETS
from .baseline import CONFIG, MANIFEST_KEY, MANIFEST_SHA, PREFIX, SPLIT, save_once, sequences

L0_SOURCE = "3eeb0c747927a8931ada80643fe3e83dd80245ba"
LEASE_SECONDS = 24*3600
GEOMETRY_SOURCE = "1574ebb51fb5ca148d1fd985821c81b95bf76548"


def read(client, bucket, key):
    response = client.get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        data = body.read()
    if sha256(data).hexdigest() != response["Metadata"].get("sha256"):
        raise ValueError("Private receipt digest mismatch")
    return json.loads(data)


def has(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as ex:
        if ex.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            return False
        raise
    return True


def get_grant(client, bucket, key):
    head = client.head_object(Bucket=bucket, Key=key)
    digest = head["Metadata"]["sha256"]
    return dict(url=client.generate_presigned_url("get_object",
                Params=dict(Bucket=bucket, Key=key), ExpiresIn=LEASE_SECONDS),
                sha256=digest, suffix=".nii.gz" if key.endswith(".gz") else ".nii")


def grant_output(writer, bucket, key):
    def sign(method, path, extra):
        return writer.generate_presigned_url(method,
                Params=dict(Bucket=bucket, Key=path, **extra), ExpiresIn=LEASE_SECONDS)
    tag = {"IfNoneMatch": "*"}
    return dict(put=sign("put_object", key, tag), get=sign("get_object", key, {}),
                receipt=sign("put_object", key+".receipt.json", tag))


def archive_key(prefix, mode, index):
    return f"{prefix}/{mode}/unit-{index:02}.tar.gz"


def assemble(root, reader, canonical, writer, output, source, active, seconds):
    path = root/"canonical.jsonl"
    reader.download_file(canonical, MANIFEST_KEY, str(path))
    data = Dataset(path, SPLIT, root/"cache", partition="validation")
    groups = list(sequences(data.visits).values())
    if (len(groups), sum(map(len, groups)), sum(len(g)-1 for g in groups)) != (45, 159, 114):
        raise ValueError("Unexpected validation inventory")
    prefix = f"{PREFIX}/tracking/{source}/{active['checkpoint_sha256']}"
    l0_config = sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
    l0_dir = f"{PREFIX}/{MANIFEST_SHA}/{L0_SOURCE}/{l0_config}"
    expires = int(time.time()+LEASE_SECONDS-300)

    def item(order, visits):
        l0 = f"{l0_dir}/patient-{order:02}.json"
        baseline = read(writer, output, l0)
        if baseline["source_sha"] != L0_SOURCE or baseline["manifest_sha256"] != MANIFEST_SHA:
            raise ValueError("Unexpected accepted L0 provenance")
        info = baseline["visit_geometry"][0]["instances"]
        group = []
        for row in visits:
            keys = (row["modalities"]["t1c"], row["segmentation"])
            entry = {f: get_grant(reader, canonical, f"{DATASET}/raw/{p}")
                     for f,p in zip(("image","seg"),keys)}
            group.append(dict(exam=row["exam_id"], **entry))
        return dict(index=order, visits=group, l0=get_grant(writer, output, l0)), info

    with ThreadPoolExecutor(max_workers=8) as pool:
        prepared = list(pool.map(lambda x: item(*x), enumerate(groups)))
    tiny = min((min(r["volume_mm3"] for r in info), i)
               for i, (_, info) in enumerate(prepared) if info)[1]
    multi = next(i for i, (_, info) in enumerate(prepared) if len(info) > 1 and i != tiny)
    pilot = [tiny, multi]
    order = pilot+[i for i in range(45) if i not in pilot]

    def jobs(mode, indices):
        available = []
        for index in indices:
            key = archive_key(prefix, mode, index)
            if has(writer, output, key+".receipt.json"):
                continue
            if has(writer, output, key):
                raise ValueError("Partial private commit requires independent recovery before resumption")
            work = dict(prepared[index][0], output=grant_output(writer, output, key))
            available.append(work)
        return available

    materials = {name.split("/", 1)[1]: get_grant(writer, output, f"{ASSETS}/{name}")
                 for name in MEMBERS}
    for path, field in (("dataset.json","dataset_sha256"), ("plans.json","plans_sha256"),
                        ("fold_0/checkpoint_final.pth", "checkpoint_sha256")):
        if materials[path]["sha256"] != active[field]:
            raise ValueError("Asset grant identity differs from inspected configuration")
    package = dict(campaign=prefix, source_sha=source, expires=expires, session_seconds=seconds,
                   assets=materials, runtime=active, smoke=jobs("smoke", pilot),
                   geometry=jobs("geometry", order), formal=jobs("formal", order),
                   pilot_size=len(pilot), evaluation_partition="validation")
    return package


def cli(*args, acceptable=()):
    call = subprocess.run(["kaggle",*args], capture_output=True, text=True)
    if call.returncode and not any(msg.lower() in (call.stdout+call.stderr).lower() for msg in acceptable):
        raise RuntimeError(f"Kaggle {args[0]} {args[1]} failed ({call.returncode})")
    return call


def publish_input(root, package, api, dataset):
    folder = root/"private-input"
    folder.mkdir()
    body = json.dumps(package, separators=(",", ":"), sort_keys=True).encode()
    digest = sha256(body).hexdigest()
    (folder/"lesionlocator.json").write_bytes(body)
    (folder/"dataset-metadata.json").write_text(json.dumps(dict(
        id=dataset, title="Novo LesionLocator private execution grants",
        licenses=[{"name":"other"}], isPrivate=True)))
    matches = api.dataset_list(mine=True, search=dataset.split("/")[1])
    absent = not any(d.ref == dataset for d in matches)
    if absent:
        cli("datasets","create","-q","-p",str(folder))
    else:
        cli("datasets","version","-q","-p",str(folder),"-m","Immutable LesionLocator package "+digest[:12])
    recovery = root/"independent-input"
    recovery.mkdir()
    for _ in range(36):
        try:
            api.dataset_download_files(dataset, path=str(recovery), force=True, quiet=True)
            with ZipFile(next(recovery.glob("*.zip"))) as archive:
                if sha256(archive.read("lesionlocator.json")).hexdigest() == digest:
                    print(json.dumps(dict(event="private_kaggle_input_readback", sha256=digest,
                                          expires_utc=package["expires"])), flush=True)
                    return digest
        except Exception:
            pass
        time.sleep(10)
    raise RuntimeError("Exact private Kaggle input did not become readable")


def submit(root, source, dataset, kernel, digest, mode, seconds):
    folder = root/mode
    folder.mkdir()
    code = Path(__file__).with_name("kaggle_bootstrap.py").read_text()
    exact = f"SOURCE_SHA={source!r}\nEXPECTED_PACKAGE={digest!r}\nMODE={mode!r}\n"
    (folder/"main.py").write_text(exact+code)
    (folder/"kernel-metadata.json").write_text(json.dumps(dict(
        id=kernel, title=kernel.split("/")[-1].replace("-"," "), code_file="main.py",
        language="python", kernel_type="script", is_private=True, enable_gpu=mode=="gpu",
        enable_internet=True, dataset_sources=[dataset], competition_sources=[], kernel_sources=[])))
    args = ["kernels","push","-p",str(folder),"-t",str(seconds)]
    if mode == "gpu":
        args += ["--accelerator","NvidiaTeslaT4"]
    result = cli(*args)
    if "version" not in result.stdout.lower():
        raise RuntimeError("Kaggle submission acknowledgement missing; verify before retry")
    print(json.dumps(dict(event="submission_acknowledged", mode=mode, kernel=kernel)), flush=True)


def collect(client, output, prefix, source):
    aggregates = {}
    for mode in ("smoke", "geometry", "formal"):
        diagnostics = Diagnostics()
        stats = dict(patients=0, exams=0, pairs=0, seconds=0., l1_dice_sum=0.,
                     l0_dice_sum=0., warp_dice_sum=0., normalized_visits=0, instances=0,
                     preprocessing_secs=0., peak_vram=0, peak_ram=0,
                     registration_seconds=0., unet_seconds=0., export_seconds=0., read_seconds=0.)
        for i in range(45):
            key = archive_key(prefix, mode, i)
            if not has(client, output, key+".receipt.json"):
                continue
            with client.get_object(Bucket=output, Key=key+".receipt.json")["Body"] as body:
                receipt = json.load(body)
            with tempfile.TemporaryFile() as archive:
                client.download_fileobj(output, key, archive)
                if archive.tell() != receipt["bytes"]:
                    raise ValueError("Recovered archive byte count mismatch")
                archive.seek(0)
                if file_digest(archive, "sha256").hexdigest() != receipt["sha256"]:
                    raise ValueError("Independent archive checksum mismatch")
                archive.seek(0)
                with tarfile.open(fileobj=archive, mode="r:gz") as files:
                    report = json.load(files.extractfile("report.json"))
                    for item in files.getmembers():
                        if item.name.endswith("/trace.json"):
                            t = json.load(files.extractfile(item))
                            m = t["resources"]
                            stats["peak_ram"] = max(m["observed_rss_bytes"], stats["peak_ram"])
                            stats["peak_vram"] = max(m["peak_cuda_allocated_bytes"], stats["peak_vram"])
                            stats["preprocessing_secs"] += sum(p["seconds"] for p in t.get("T1", []))
                            stats["registration_seconds"] += t.get("cuda_seconds", {}).get("registration", 0.)
                            stats["unet_seconds"] += t.get("cuda_seconds", {}).get("unet", 0.)
                            stats["export_seconds"] += t.get("native_export_seconds", 0.)
            if report["source_sha"] != source or report["campaign"] != prefix or report["index"] != i:
                raise ValueError("Recovered result provenance mismatch")
            diagnostics.add(report)
            stats["patients"] += 1
            stats["exams"] += report["exams"]
            stats["pairs"] += len(report["pairs"])
            stats["seconds"] += report["total_seconds"]
            stats["read_seconds"] += report["io_seconds"]
            stats["normalized_visits"] += len(report["normalization"])
            for pair in report["pairs"]:
                stats["l0_dice_sum"] += pair["L0"]["dice"]
                if "L1" in pair:
                    l1 = pair["L1"]
                    stats["l1_dice_sum"] += l1["union"]["dice"]
                    stats["warp_dice_sum"] += l1["warp_union"]["dice"]
                    stats["instances"] += len(l1["instances"])
        aggregates[mode] = dict(stats, diagnostics=diagnostics.result())
    print(json.dumps(dict(event="independent_archive_readback", source=source, results=aggregates)), flush=True)
    if os.getenv("GITHUB_RUN_ID"):
        save_once(client, output, prefix+f"/reports/collector-{os.environ['GITHUB_RUN_ID']}.json", aggregates)


def main():
    source, mode = os.environ["NOVO_SOURCE_SHA"], os.environ["MODE"]
    reader, canonical = connect()
    writer, output = connect(output=True)
    if (canonical, output) != ("novo-nordisk-pbl","novo-nordisk-pbl-outputs"):
        raise PermissionError("Expected private project buckets")
    if not has(writer, output, ASSETS+"/runtime.json") and mode in ("prepare", "benchmark"):
        from .inspect_checkpoint import main as inspect_checkpoint
        inspect_checkpoint()
    active = read(writer, output, ASSETS+"/runtime.json")
    prefix = f"{PREFIX}/tracking/{source}/{active['checkpoint_sha256']}"
    if mode in ("collect","recover"):
        collect(writer, output, prefix, source)
        if source != GEOMETRY_SOURCE:
            geometry_prefix = f"{PREFIX}/tracking/{GEOMETRY_SOURCE}/{active['checkpoint_sha256']}"
            collect(writer, output, geometry_prefix, GEOMETRY_SOURCE)
        return
    if mode not in ("prepare","benchmark"):
        raise ValueError("Expected an inference-only bridge mode")
    api = KaggleApi()
    api.authenticate()
    seconds = int(os.environ["GPU_SESSION_SECONDS"])
    if mode == "benchmark":
        if os.environ["FREE_RESOURCE_AUTHORIZED"] != "true":
            raise PermissionError("The registered job lacks free GPU authorization")
        quota = api.quota_view().gpu_quota
        if quota.is_pay_to_scale_enabled:
            raise PermissionError("This submission requires free-only Kaggle allocation")
        available = (quota.total_time_allowed - quota.time_used - quota.time_reserved).total_seconds()
        seconds = min(seconds, int(available//2)-300)
        if seconds < 1200:
            raise RuntimeError("Insufficient remaining free GPU quota")
        print(json.dumps(dict(free_gpu_quota_seconds=available, session_seconds=seconds)))
    with tempfile.TemporaryDirectory(prefix="novo-ll-publication-") as d:
        root = Path(d)
        package = assemble(root, reader, canonical, writer, output, source, active, seconds)
        dataset, base = os.environ["DATASET"], os.environ["KERNEL"]
        if mode == "prepare":
            publish_input(root, package, api, dataset)
            return
        run = os.environ["GITHUB_RUN_ID"]
        permit = prefix+f"/submission-{run}.json"
        if has(writer, output, permit):
            raise RuntimeError("Existing submission reservation; inspect live kernels before resumption")
        digest = publish_input(root, package, api, dataset)
        # Same workflow run ID cannot submit twice even if an acknowledgement is lost.
        save_once(writer, output, permit,
                  dict(package_sha256=digest, expires=package["expires"], session_seconds=seconds))
        if package["geometry"] and source == GEOMETRY_SOURCE:
            submit(root, source, dataset, base+f"-g-{run}", digest, "geometry", min(seconds, 5400))
        if package["formal"]:
            submit(root, source, dataset, base+f"-t-{run}", digest, "gpu", seconds)


if __name__ == "__main__":
    main()
