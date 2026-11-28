"""Audit and aggregate original private results without re-running model inference."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
from math import sqrt
import os
from statistics import mean, median
import tarfile
import tempfile

from cloud.storage import connect, put_once

ROOT = "novo-nordisk/experiments/lesionlocator"
MANIFEST = "040ec6c523d5beae43a30b64fede2bee226e1a08e8c14b2cca8043b01b9c39f4"
BASELINE = "3eeb0c747927a8931ada80643fe3e83dd80245ba"
GEOMETRY = "1574ebb51fb5ca148d1fd985821c81b95bf76548"
GPU = "9fff1e41eb6a22000c6fd04984dbced6e05ae96f"
CHECKPOINT = "f00f2e71392a88f922fcdfa852b71c9fdbdf9c14c33472a21c104398f2266596"
L0_CONFIG = {"condition": "L0", "native": "T1c", "target": 3,
             "connectivity": 26, "interpolation_order": 0,
             "identity_reference": "same-grid conservative unique mutual overlap of native GT only"}
TAGS = ("L0","L1-Warp","L1-Final","L2")


def read(client, bucket, key, check=True):
    response = client.get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        data = body.read()
    if check and response["Metadata"].get("sha256") != sha256(data).hexdigest():
        raise ValueError("Source JSON integrity mismatch")
    return json.loads(data)


def recover(client, bucket, key, roster, index, mode, source):
    receipt = read(client,bucket,key+".receipt.json",False)
    expected = roster[index]
    with tempfile.TemporaryFile() as blob:
        client.download_fileobj(bucket,key,blob)
        if blob.seek(0,os.SEEK_END) != receipt["bytes"]:
            raise ValueError("Archive size mismatch")
        blob.seek(0)
        if file_digest(blob,"sha256").hexdigest() != receipt["sha256"]:
            raise ValueError("Archive SHA-256 mismatch")
        blob.seek(0)
        with tarfile.open(fileobj=blob,mode="r:gz") as tar:
            data = json.load(tar.extractfile("report.json"))
            traces = {item.name:json.load(tar.extractfile(item))
                      for item in tar if item.name.endswith("/trace.json")}
    if (data["index"],data["source_sha"],data["checkpoint_sha256"],data["smoke"]) != (
            index,source,CHECKPOINT,mode=="smoke"):
        raise ValueError("Unexpected result provenance")
    pairs = expected["direct_l0"][:1] if mode=="smoke" else expected["direct_l0"]
    if [p["L0"] for p in data["pairs"]] != [p["metrics"] for p in pairs]:
        raise ValueError("Pair inventory or frozen baseline disagreement")
    if data["exams"] != len(pairs)+1 or len(data["normalization"]) != data["exams"]:
        raise ValueError("Unexpected visit denominator")
    return data,traces,receipt


def avg(values):
    return mean(values) if values else None


def union(data):
    rows=[r for patient in data for r in patient]
    positive=[r for r in rows if r["truth_lesions"]]
    matched=sum(r["matched_lesions"] for r in rows)
    truth=sum(r["truth_lesions"] for r in rows)
    fp=sum(r["false_positives"] for r in rows)
    return dict(pairs=len(rows), ground_truth_positive=len(positive), matched=matched, truth=truth,
                fp=fp, dice=avg([r["dice"] for r in rows]),
                positive_dice=avg([r["dice"] for r in positive]),
                patient_macro_dice=avg([mean([r["dice"] for r in patient]) for patient in data if patient]),
                fp_per_pair=fp/len(rows) if rows else None,
                empty_gt_fp=sum(r["false_positives"] for r in rows if not r["truth_lesions"]),
                positive_gt_fp=sum(r["false_positives"] for r in positive),
                sensitivity=matched/truth if truth else None,
                hd95_mm=avg([r["hd95_mm"] for r in positive if r["hd95_mm"] is not None]),
                hd95_positive_defined=sum(r["hd95_mm"] is not None for r in positive),
                abs_volume_error_mm3=avg([r["absolute_volume_error_mm3"] for r in rows]))


def aggregate(records, mode):
    patient_metrics=defaultdict(list)
    geometry=defaultdict(list)
    norm=defaultdict(list)
    contrast=[]
    stages=Counter()
    sources=Counter()
    timing=Counter()
    roi=defaultdict(list)
    local=defaultdict(list)
    proxy=defaultdict(list)
    l2=Counter()
    versions=set()
    device=Counter()
    for report,traces,receipt in records:
        per_patient=defaultdict(list)
        if mode!="geometry":
            sources.update(report["pairs"][0]["L2"]["initial_ids"] and ["initial_gt_present"] or ["no_initial_gt"])
        device[report["device"]]+=1
        versions.add(json.dumps(report.get("versions"),sort_keys=True))
        timing.update(worker_seconds=report["total_seconds"], read_seconds=report["io_seconds"],
                      verified_bytes=receipt["bytes"])
        for scan in report["normalization"]:
            for group in ("full","nonzero","lesion","nonzero_perilesional_shell"):
                if scan.get(group,{}).get("n"):
                    norm[group].append(scan[group])
            if scan.get("signed_mean_contrast"):
                contrast.append(scan["signed_mean_contrast"])
        for pair in report["pairs"]:
            per_patient["L0"].append(pair["L0"])
            for stage,values in (pair["geometry"] or {}).items():
                geometry[stage].extend(values["instances"])
            if "L1" not in pair:
                continue
            direct, ar=pair["L1"],pair["L2"]
            per_patient["L1-Warp"].append(direct["warp_union"])
            per_patient["L1-Final"].append(direct["union"])
            per_patient["L2"].append(direct["union"] if ar.get("shared_initial_l1_output") else ar["union"])
            if ar.get("shared_initial_l1_output"):
                l2.update(shared_initial_pairs=1, initial_ids=len(ar["initial_ids"]),
                          initial_lost=len(ar["missing_after_step"]))
            else:
                l2["later_pairs"]+=1
                l2.update("later_"+r["status"] for r in ar["instances"])
                l2["later_nonempty_pairs"]+=int(ar["union"]["matched_lesions"]+ar["union"]["false_positives"]>0)
            for r in direct["instances"]:
                stages["L1_status_"+r["status"]]+=1
                for k,v in r.get("T6",{}).items():
                    if v is not None:
                        roi[k].append(v)
                e=r.get("T7",{}).get("union_gt")
                if e is not None:
                    stages["T7_scored"]+=1
                    stages["T7_positive"]+=int(e["predicted_voxels"]>0)
                    if e["dice"] is not None:
                        local["T7_roi_union_dice"].append(e["dice"])
                e=r.get("T8",{}).get("work_vs_final_on_native")
                if e is not None:
                    stages["T8_pre_native_positive"]+=int(e["reference_voxels"]>0)
                    stages["T8_native_positive"]+=int(e["predicted_voxels"]>0)
                    stages["T8_pre_positive_final_empty"]+=int(e["reference_voxels"]>0 and e["predicted_voxels"]==0)
                for tag,k in (("L0","L0"),("L1-Warp","warp"),("L1-Final","final")):
                    p=(r[k] or {}).get("reference_proxy")
                    if p is not None:
                        proxy[tag].append(p["dice"])
        for arm,items in per_patient.items():
            patient_metrics[arm].append(items)
        for name,t in traces.items():
            if "/L1/" in name:
                stages["T0_trace"]+=1
                history=next((m["prompt_voxels"] for m in t["T1"] if m["prompt_voxels"] is not None),0)
                stages["T1_prompt_empty"]+=int(history==0)
                if "T2" in t:
                    stages["T2_measured"]+=1
                    stages["T2_cropped"]+=int(t["T2"]["before"]!=t["T2"]["after"])
                if "T3" in t:
                    stages["T3_measured"]+=1
                    stages["T3_prompt_empty"]+=int(t["T3"]["nonzero"]==0)
                    stages["T4_field_finite"]+=int(t["T4"]["finite"])
                    stages["T5_grid_empty"]+=int(t["T5_grid"]["nonzero"]==0)
                    stages["T5_native_empty"]+=int(t["T5_native_voxels"]==0)
                    stages["T6_padding_axis_fix"]+=int(t["T6"]["axis_compatibility_fix"])
            res=t["resources"]
            timing["peak_cuda"]=max(timing["peak_cuda"],res["peak_cuda_allocated_bytes"])
            timing["peak_rss"]=max(timing["peak_rss"],res["observed_rss_bytes"])
            timing["peak_cuda_reserved"]=max(timing["peak_cuda_reserved"],res.get("peak_cuda_reserved_bytes",0))
            timing.update(preprocessing_seconds=sum(m["seconds"] for m in t["T1"]),
                          registration_seconds=t.get("cuda_seconds",{}).get("registration",0),
                          unet_seconds=t.get("cuda_seconds",{}).get("unet",0),
                          export_seconds=t.get("native_export_seconds",0))
    gstats={}
    for stage,rows in geometry.items():
        gstats[stage]=dict(instances=len(rows), lost=sum(not r["exists"] for r in rows),
            dice=avg([r["dice"] for r in rows if r.get("dice") is not None]),
            mean_abs_volume_error_mm3=avg([abs(r["volume_change_mm3"]) for r in rows
                                          if "volume_change_mm3" in r]),
            median_centroid_shift_mm=(median(shifts) if (shifts := [
                sqrt(sum(v*v for v in r["centroid_delta_mm"])) for r in rows
                if r.get("centroid_delta_mm") is not None]) else None))
    nstats={}
    for group,rows in norm.items():
        n=sum(r["n"] for r in rows)
        nstats[group]=dict(voxels=n,visits=len(rows), upper=sum(r["n"]*r["high_clip"] for r in rows)/n,
                          lower=sum(r["n"]*r["low_clip"] for r in rows)/n,
                          median_visit_raw_median=median(r["quantiles"][2] for r in rows),
                          range_visit_raw_median=[min(r["quantiles"][2] for r in rows),
                                                  max(r["quantiles"][2] for r in rows)])
    if contrast:
        nstats["contrast"]=dict(visits=len(contrast),
           median_abs_raw=median(abs(r["raw_mean"]) for r in contrast),
           median_abs_normalized=median(abs(r["normalized_mean"]) for r in contrast))
    return dict(patients=len(records), exams=sum(r["exams"] for r,_,_ in records),
                pairs=sum(len(r["pairs"]) for r,_,_ in records), device=dict(device),
                versions=[json.loads(v) for v in sorted(versions)], timing=dict(timing),
                arms={k:union(v) for k,v in patient_metrics.items()},geometry=gstats,
                normalization=nstats, stages=dict(stages), initial=dict(sources), l2=dict(l2),
                roi={k:dict(n=len(v),mean=mean(v)) for k,v in roi.items()},
                proxy={k:dict(n=len(v),dice=mean(v)) for k,v in proxy.items()},
                local={k:dict(n=len(v),dice=mean(v)) for k,v in local.items()})


def main():
    c,b=connect(output=True)
    if b!="novo-nordisk-pbl-outputs":
        raise PermissionError("Expected private project output storage")
    d=sha256(json.dumps(L0_CONFIG,sort_keys=True).encode()).hexdigest()
    roster=[read(c,b,f"{ROOT}/{MANIFEST}/{BASELINE}/{d}/patient-{i:02}.json") for i in range(45)]
    if sum(len(r["visit_geometry"]) for r in roster)!=159 or sum(len(r["direct_l0"]) for r in roster)!=114:
        raise ValueError("Unexpected frozen denominator")
    summary, inventory={}, {}
    for mode,source in (("smoke",GPU),("formal",GPU),("geometry",GEOMETRY)):
        prefix=f"{ROOT}/tracking/{source}/{CHECKPOINT}/{mode}/"
        pages=c.get_paginator("list_objects_v2").paginate(Bucket=b,Prefix=prefix)
        names={o["Key"] for page in pages for o in page.get("Contents",[])}
        if any(k not in {f"{prefix}unit-{i:02}.tar.gz{suffix}" for i in range(45)
                         for suffix in ("",".receipt.json")} for k in names):
            raise ValueError("Unexpected archive slot")
        index_to_key={i:f"{prefix}unit-{i:02}.tar.gz" for i in range(45)}
        committed=[i for i,k in index_to_key.items() if k in names and k+".receipt.json" in names]
        orphans=[i for i,k in index_to_key.items() if (k in names)!=(k+".receipt.json" in names)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            records=list(pool.map(lambda i: recover(c,b,index_to_key[i],roster,i,mode,source),committed))
        stats=aggregate(records,mode)
        stats["orphans"]=len(orphans)
        stats["pending_patients"]=45-len(committed) if mode!="smoke" else 2-len(committed)
        summary[mode]=stats
        inventory[mode]=dict(committed=[dict(index=i, **r[2]) for i,r in zip(committed,records)],
            pending=[i for i in range(45) if i not in committed] if mode!="smoke" else [],
            orphan_slots=orphans)
        print(json.dumps(dict(event="final_archive_audit",mode=mode,**stats),allow_nan=False),flush=True)
    with tempfile.NamedTemporaryFile() as f:
        f.write(json.dumps(dict(summary=summary,inventory=inventory),allow_nan=False).encode())
        f.flush()
        from pathlib import Path
        result=put_once(c,b,f"{ROOT}/reports/terminal-{os.environ['GITHUB_RUN_ID']}.json",Path(f.name))
    print(json.dumps(dict(event="private_terminal_receipt",sha256=result["sha256"])),flush=True)


if __name__=="__main__":
    main()
