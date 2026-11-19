"""Fixed .5 control metrics; 26-connectivity and shared one-to-one overlap matching."""
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from research.evaluate import components

STRATA=('lt100_mm3','100_to_lt1000_mm3','ge1000_mm3')

def lesion_strata(prediction,truth,affine):
    p,ps=components(prediction,affine);t,ts=components(truth,affine)
    overlap=np.zeros((len(ts),len(ps)),bool);mask=truth & prediction
    if mask.any():
        pairs=np.unique(np.stack([t[mask],p[mask]],axis=1),axis=0)
        overlap[pairs[:,0]-1,pairs[:,1]-1]=True
    matched=set()
    if overlap.size:
        rows,cols=linear_sum_assignment(-overlap.astype(int));matched={int(r) for r,c in zip(rows,cols) if overlap[r,c]}
    result={s:dict(truth=0,matched=0) for s in STRATA}
    for i,r in enumerate(ts):
        s=STRATA[0 if r['volume_mm3']<100 else 1 if r['volume_mm3']<1000 else 2]
        result[s]['truth']+=1;result[s]['matched']+=int(i in matched)
    return result

def control_metrics(visits,complete):
    if not complete:return dict(complete=False)
    groups=defaultdict(list)
    for r in visits:groups[r['patient_id']].append(r)
    means=lambda xs:float(np.mean(xs)) if xs else None
    dice=[];sens=[];fp=[];strata={s:[] for s in STRATA};empty=[];positive=[]
    for rows in groups.values():
        pos=[r for r in rows if r['truth_lesions']>0]
        if pos:
            dice.append(means([r['dice'] for r in pos]))
            sens.append(means([r['lesion_sensitivity'] for r in pos]))
        fp.append(means([r['false_positives'] for r in rows]))
        for s in STRATA:
            total=sum(r['lesion_strata'][s]['truth'] for r in rows)
            if total:strata[s].append(sum(r['lesion_strata'][s]['matched'] for r in rows)/total)
        for flag,target in [(True,positive),(False,empty)]:
            values=[r['false_positives'] for r in rows if bool(r['truth_lesions'])==flag]
            if values:target.append(means(values))
    planes=sum(r['loss_planes'] for r in visits)
    return dict(complete=True,positive_patient_dice=means(dice),positive_dice_defined_patients=len(dice),patient_lesion_sensitivity=means(sens),sensitivity_defined_patients=len(sens),patient_fp_per_scan=means(fp),small_lesion_sensitivity={s:means(v) for s,v in strata.items()},small_lesion_defined_patients={s:len(v) for s,v in strata.items()},empty_patient_fp_per_scan=means(empty),positive_patient_fp_per_scan=means(positive),validation_loss=sum(r['loss_sum'] for r in visits)/planes,loss_planes=planes,threshold_probability=.5)
