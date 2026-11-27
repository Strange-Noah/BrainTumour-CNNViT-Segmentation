"""Public aggregates without examination keys, masks, or individual intensities."""
from collections import defaultdict


class Diagnostics:
    def __init__(self):
        self.geometry = defaultdict(lambda: dict(instances=0, lost=0, dice_sum=0., scored=0))
        self.normalization = defaultdict(lambda: dict(voxels=0, below=0., above=0.))
        self.roi = dict(scored=0, coverage_sum=0., reference_scored=0, reference_coverage_sum=0.)
        self.reference = defaultdict(lambda: dict(scored=0, dice_sum=0.))
        self.metrics = defaultdict(lambda: dict(pairs=0, matched=0, truth=0, fp=0, dice_sum=0.))

    def add(self, report):
        for sample in report["normalization"]:
            for label in ("full","nonzero","lesion"):
                group = sample.get(label, {})
                n = group.get("n",0)
                if n:
                    aggregate = self.normalization[label]
                    aggregate["voxels"] += n
                    aggregate["below"] += group["low_clip"]*n
                    aggregate["above"] += group["high_clip"]*n
        for pair in report["pairs"]:
            for tag, arm in (pair.get("geometry") or {}).items():
                for row in arm["instances"]:
                    agg = self.geometry[tag]
                    agg["instances"] += 1
                    agg["lost"] += int(not row["exists"])
                    if row.get("dice") is not None:
                        agg["dice_sum"] += row["dice"]
                        agg["scored"] += 1
            model = pair.get("L1")
            if model is None:
                continue
            for name, result in (("L0",pair["L0"]),("L1-Warp",model["warp_union"]),
                                 ("L1-Final",model["union"])):
                aggregate = self.metrics[name]
                aggregate["pairs"] += 1
                aggregate["matched"] += result["matched_lesions"]
                aggregate["truth"] += result["truth_lesions"]
                aggregate["fp"] += result["false_positives"]
                aggregate["dice_sum"] += result["dice"]
            for lesion in model["instances"]:
                for tag, src in (("L0",lesion["L0"]),("L1-Warp",lesion["warp"]),("L1-Final",lesion["final"])):
                    proxy = (src or {}).get("reference_proxy")
                    if proxy is not None:
                        self.reference[tag]["scored"] += 1
                        self.reference[tag]["dice_sum"] += proxy["dice"]
                roi = lesion.get("T6", {})
                if roi.get("union_gt_coverage") is not None:
                    self.roi["scored"] += 1
                    self.roi["coverage_sum"] += roi["union_gt_coverage"]
                if roi.get("reference_coverage") is not None:
                    self.roi["reference_scored"] += 1
                    self.roi["reference_coverage_sum"] += roi["reference_coverage"]

    def result(self):
        return dict(geometry=dict(self.geometry), normalization=dict(self.normalization),
                    roi=self.roi, reference_proxy=dict(self.reference), spatial_metrics=dict(self.metrics))
