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

def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan).encode()).hexdigest()

def exhaustive_plan(visits,seed=42):
    if any(v['partition']!='train' for v in visits):raise ValueError('Training visits only')
    rng=np.random.default_rng(seed);groups=defaultdict(list)
    for v in visits:groups[v['metadata']['patient_id']].append(v['metadata'])
    patients=sorted(groups);rng.shuffle(patients);plan=[]
    for patient in patients:
        rows=sorted(groups[patient],key=lambda m:m['exam_id']);rng.shuffle(rows)
        for m in rows:
            for t in sorted(m['tiles'],key=lambda t:tuple(t['origin'])):
                zs=sorted(t['positive']+t['negative'])
                if zs!=list(range(m['shape'][2])):raise ValueError('Incomplete tile-plane inventory')
                plan.extend((m['exam_id'],t['origin'],z,'exhaustive') for z in zs)
    if len({(e,tuple(t),z) for e,t,z,_ in plan})!=len(plan):raise ValueError('Duplicate positions')
    if len(plan)%2:raise ValueError('Exhaustive DDP plan cannot pad or drop positions')
    return plan,plan_digest(plan)

def adaptive_plan(visits,cycle,ratios,difficulty,step,total=67200,seed=42):
    """Freeze the draw multiset, then group for locality. Caps apply to all categories."""
    from .adaptive import CATEGORIES
    if any(v['partition']!='train' for v in visits):raise ValueError('Training only')
    if ratios['baseline']<.3-1e-9 or abs(sum(ratios.values())-1)>1e-8:raise ValueError('Invalid mixture')
    rng=np.random.default_rng(np.random.SeedSequence([seed,cycle,2]));meta={v['metadata']['exam_id']:v['metadata'] for v in visits}
    by_patient=defaultdict(list)
    for e,m in meta.items():by_patient[m['patient_id']].append(e)
    patients=sorted(by_patient);counts=defaultdict(int);vcounts=defaultdict(int)
    patient_cap=max(int(total*.01),int(np.ceil(total/len(patients))))
    visit_cap=max(int(total*.005),int(np.ceil(total/len(meta))))
    mined=defaultdict(list)
    for r in difficulty.records.values():mined[r['exam'],r['kind']].append(r)
    budgets={k:int(total*ratios[k]) for k in CATEGORIES};budgets['baseline']+=total-sum(budgets.values())
    plan=[];fallback=defaultdict(int)
    # Baseline first has an exact patient-balanced anchor (128 each at 40%).
    for category in CATEGORIES:
        for _ in range(budgets[category]):
            eligible=[]
            for p in patients:
                if counts[p]>=patient_cap:continue
                exams=[]
                for e in by_patient[p]:
                    if vcounts[e]>=visit_cap:continue
                    key='positive' if category=='hard_positive' else 'negative'
                    if category=='baseline' or category=='hard_negative' or any(t[key] for t in meta[e]['tiles']):exams.append(e)
                if exams:eligible.append((p,exams))
            if not eligible:raise ValueError('Mixture infeasible under patient/visit caps')
            lowest=min(counts[p] for p,_ in eligible);eligible=[x for x in eligible if counts[x[0]]==lowest]
            p,exams=eligible[int(rng.integers(len(eligible)))];e=exams[int(rng.integers(len(exams)))];m=meta[e]
            kind='positive' if category=='hard_positive' else 'negative';records=mined[e,kind] if category.startswith('hard_') else []
            if records:
                weights=np.array([difficulty.weight(r,step) for r in records]);r=records[int(rng.choice(len(records),p=weights/weights.sum()))];tile,z=r['tile'],r['z']
            else:
                if category.startswith('hard_'):fallback[category]+=1
                if category=='baseline':kind='positive' if rng.random()<.5 else 'negative'
                options=[t for t in m['tiles'] if t[kind]]
                if not options:kind='negative' if kind=='positive' else 'positive';options=[t for t in m['tiles'] if t[kind]]
                t=options[int(rng.integers(len(options)))];tile=t['origin'];z=int(rng.choice(t[kind]))
            plan.append((e,tile,z,category));counts[p]+=1;vcounts[e]+=1
    order=list(patients);rng.shuffle(order);priority={p:i for i,p in enumerate(order)}
    plan.sort(key=lambda d:(priority[meta[d[0]]['patient_id']],d[0],tuple(d[1]),d[2]//8,d[2]))
    return plan,plan_digest(plan),dict(category_counts=budgets,fallback_counts=dict(fallback),patient_cap=patient_cap,visit_cap=visit_cap,max_patient=max(counts.values()),max_visit=max(vcounts.values()))
