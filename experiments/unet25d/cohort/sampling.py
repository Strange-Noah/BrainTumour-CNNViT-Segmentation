"""Coverage-first frozen cohort draws; serialize epoch/global cursor and plan SHA."""
import hashlib,json
from collections import defaultdict,OrderedDict
import numpy as np

def epoch_plan(visits,epoch,seed=42,samples_per_patient=64):
    if any(v['partition']!='train' for v in visits):raise ValueError('No validation/test training draws')
    patients=sorted({v['metadata']['patient_id'] for v in visits});rng=np.random.default_rng(np.random.SeedSequence([seed,epoch]));coverage=[];pools={True:defaultdict(list),False:defaultdict(list)}
    for v in sorted(visits,key=lambda v:v['metadata']['exam_id']):
        m=v['metadata']
        for tile in m['tiles']:
            for positive,key in [(True,'positive'),(False,'negative')]:
                if tile[key]:pools[positive][m['patient_id']].append((m['exam_id'],tile['origin'],tile[key]))
            choices=tile['positive'] or tile['negative']
            coverage.append((m['exam_id'],tile['origin'],int(rng.choice(choices)),bool(tile['positive'])))
    total=len(patients)*samples_per_patient
    if len(coverage)>total//4 or not all(pools.values()):raise ValueError('Coverage quota/class pools need an explicit configuration decision')
    enhancing={v['metadata']['exam_id']:any(t['positive'] for t in v['metadata']['tiles']) for v in visits}
    draws=list(coverage)
    for stratum,budget in [('positive',total//2),('negative-enhancing-visit',total//4),('negative-empty-visit',total//4)]:
        flag=stratum=='positive';wanted=stratum=='negative-enhancing-visit'
        candidates={p:[x for x in rows if flag or enhancing[x[0]]==wanted] for p,rows in pools[flag].items()};candidates={p:rows for p,rows in candidates.items() if rows}
        if not candidates:raise ValueError('Required frozen sampling stratum is absent')
        existing=sum(d[3] if flag else (not d[3] and enhancing[d[0]]==wanted) for d in draws)
        for _ in range(budget-existing):
            patient=str(rng.choice(sorted(candidates)));options=candidates[patient];visit=str(rng.choice(sorted({x[0] for x in options})));tiles=[x for x in options if x[0]==visit];visit,tile,zs=tiles[int(rng.integers(len(tiles)))];draws.append((visit,tile,int(rng.choice(zs)),flag))
    rng.shuffle(draws)
    # Random patient blocks per epoch preserve draws while limiting shard churn.
    visit_patient={v['metadata']['exam_id']:v['metadata']['patient_id'] for v in visits}
    order=list(patients);rng.shuffle(order);groups={p:[] for p in order}
    for d in draws:groups[visit_patient[d[0]]].append(d)
    ordered=[]
    for patient in order:ordered.extend(sorted(groups[patient],key=lambda d:(d[0],tuple(d[1]),d[2]//8,d[2])))
    digest=hashlib.sha256(json.dumps(ordered).encode()).hexdigest()
    return ordered,digest

def resume_rank(plan,cursor,rank,world=2):
    if cursor%world or rank not in range(world):raise ValueError('Global cursor/rank mismatch')
    return plan[cursor+rank::world]
