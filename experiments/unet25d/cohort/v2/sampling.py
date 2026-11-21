"""Frozen full coverage + lesion replay; update-level class/rank interleaving."""
import hashlib,json,heapq
from collections import Counter,defaultdict,deque
import numpy as np

def digest(plan):return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def lesion_weight(volume,c):return float(np.clip((c['small_weight_reference_mm3']/max(volume,1e-9))**c['small_weight_exponent'],*c['small_weight_clip']))

def mandatory(visits):
    if any(v['partition']!='train' for v in visits):raise ValueError('Training partition only')
    rows=[]
    for v in sorted(visits,key=lambda x:x['metadata']['exam_id']):
        if v['partition']!='train':raise ValueError('Training partition only')
        m=v['metadata'];enhancing=bool(m['positive'])
        for t in m['tiles']:
            pos=set(t['positive']);neg=set(t['negative'])
            if pos&neg or pos|neg!=set(range(m['shape'][2])):raise ValueError('Incomplete mandatory positions')
            for z in sorted(pos|neg):rows.append(dict(exam=m['exam_id'],patient=m['patient_id'],origin=t['origin'],z=z,scale=512,positive=z in pos,kind='mandatory',negative_category=None if z in pos else 'negative_enhancing' if enhancing else 'negative_empty'))
    if len({(r['exam'],tuple(r['origin']),r['z']) for r in rows})!=len(rows):raise ValueError('Duplicate mandatory inventory')
    return rows

def replays(inventories,total,c,rng):
    groups=defaultdict(lambda:defaultdict(list))
    for e,v in inventories.items():
        for comp in v['components']:groups[v['patient']][e].append(comp)
    if not groups:raise ValueError('No training lesion replay candidates')
    pc=Counter();vc=Counter();lc=Counter();rows=[];scales=[]
    for scale,fraction in c['replay_scales'].items():scales.extend([int(scale)]*int(total*fraction))
    scales += [512]*(total-len(scales));rng.shuffle(scales)
    active={p:{e:list(comps) for e,comps in visits.items()} for p,visits in groups.items()}
    for i in range(total):
        active={p:{e:cs for e,cs in vs.items() if cs and vc[e]<c['replay_visit_cap']} for p,vs in active.items() if pc[p]<c['replay_patient_cap']};active={p:vs for p,vs in active.items() if vs}
        if not active:raise ValueError('Replay caps infeasible; do not silently relax')
        low=min(pc[p] for p in active);ps=[p for p in sorted(active) if pc[p]==low];p=ps[int(rng.integers(len(ps)))];vs=active[p]
        low=min(vc[e] for e in vs);es=[e for e in sorted(vs) if vc[e]==low];e=es[int(rng.integers(len(es)))];cs=vs[e]
        weights=np.array([lesion_weight(x['volume_mm3'],c) for x in cs]);j=int(rng.choice(len(cs),p=weights/weights.sum()));comp=cs[j]
        z=int(rng.choice(comp['slices']));rows.append(dict(exam=e,patient=p,z=z,scale=scales[i],positive=True,kind='replay',component=comp['id'],seed=int(rng.integers(2**31)),negative_category=None))
        pc[p]+=1;vc[e]+=1;lc[e,comp['id']]+=1
        if lc[e,comp['id']]>=c['replay_lesion_cap']:cs.pop(j)
    return rows,dict(max_patient=max(pc.values()),max_visit=max(vc.values()),max_lesion=max(lc.values()),selected_lesions=len(lc),available_lesions=sum(len(cs) for vs in groups.values() for cs in vs.values()),requested_scales=dict(Counter(r['scale'] for r in rows)))

class Pool:
    def __init__(self,rows,rng,locality=8):
        self.rows=defaultdict(lambda:defaultdict(deque));self.total=Counter();self.used=Counter();self.visit_total=Counter();self.visit_used=Counter()
        self.locality=locality;self.current={};self.block_used=Counter()
        shuffled=list(rows);rng.shuffle(shuffled)
        for r in shuffled:self.rows[r['patient']][r['exam']].append(r);self.total[r['patient']]+=1;self.visit_total[r['exam']]+=1
        # Shard-local ordering occurs ONLY inside visit queues, never as a final
        # patient-sort of the global schedule.
        for p,vs in self.rows.items():
            for e,q in vs.items():self.rows[p][e]=deque(sorted(q,key=lambda r:(r['z']//8,r['z'])))
    def take(self,active,avoid,preferred=None):
        patients=[p for p,vs in self.rows.items() if vs]
        if not patients:raise ValueError('Exhausted class pool')
        candidates=[p for p in patients if p in active and p not in avoid] or [p for p in patients if p not in avoid] or patients
        p=min(candidates,key=lambda p:((self.used[p]+1)/self.total[p],p));vs=self.rows[p]
        e=self.current.get(p)
        if e not in vs or self.block_used[p]>=self.locality:
            e=min(vs,key=lambda e:((self.visit_used[e]+1)/self.visit_total[e],e));self.current[p]=e;self.block_used[p]=0
        self.block_used[p]+=1
        row=vs[e].popleft();self.used[p]+=1;self.visit_used[e]+=1
        if not vs[e]:del vs[e]
        return row

def build_plan(visits,inventories,c):
    original=mandatory(visits);n=len(original)
    if n!=c['mandatory_positions'] or n%2:raise ValueError('Frozen 220384 mandatory positions required')
    rng=np.random.default_rng(c['seed']);nr=int(np.ceil(n/c['replay_divisor']));nr+= (n+nr)%2
    extra,account=replays(inventories,nr,c,rng)
    mp=[r for r in original if r['positive']];ng=[r for r in original if not r['positive']]
    pools={'mandatory':Pool(mp,rng,c['visit_block_samples']),'replay':Pool(extra,rng,c['visit_block_samples']),'negative':Pool(ng,rng,c['visit_block_samples'])}
    positive=len(mp)+nr;updates=(n+nr)//2
    if positive>updates:raise ValueError('Pairing policy requires at most one positive per update')
    patient_total=Counter(r['patient'] for r in original+extra);used=Counter();visit_last={};previous=[None,None];order=[];pos_seen=0;replay_seen=0;active=[]
    for step in range(updates):
        if step%c['block_updates']==0:
            eligible=[p for p in patient_total if used[p]<patient_total[p]]
            active=sorted(eligible,key=lambda p:((used[p]+1)/patient_total[p],p))[:c['patient_block_size']]
        has_positive=((step+1)*positive//updates)>(step*positive//updates)
        positive_rank=pos_seen%2
        pair=[None,None]
        for rank in (positive_rank,1-positive_rank):
            want=has_positive and rank==positive_rank
            if want:
                replay=((pos_seen+1)*nr//positive)>(pos_seen*nr//positive);kind='replay' if replay else 'mandatory';pos_seen+=1;replay_seen+=int(replay)
            else:kind='negative'
            avoid={previous[rank]}
            if pair[1-rank] is not None:avoid.add(pair[1-rank]['patient'])
            preferred=visit_last.get((rank,kind));row=pools[kind].take(active,avoid,preferred)
            pair[rank]=row;used[row['patient']]+=1;visit_last[rank,kind]=row['exam'];previous[rank]=row['patient']
        order.extend(pair)
    if replay_seen!=nr:raise AssertionError('Replay mixture count')
    audit=audit_plan(order,c);audit['replay_caps']=account
    if audit['mandatory_unique']!=n or audit['positive_fraction']<c['min_positive_fraction'] or audit['negative_only_streak']['max']>c['max_negative_only_update_streak']:raise ValueError('Global exhaustive/interleaving contract failed')
    # Both rank streams must receive positive supervision across bounded windows.
    if min(audit['rank_window_min_positive_fraction'])<.20:raise ValueError('Rank supervision clustered')
    return order,digest(order),audit

def streaks(bits):
    runs=[];n=0
    for b in bits:
        if b:n+=1
        elif n:runs.append(n);n=0
    if n:runs.append(n)
    return dict(max=max(runs,default=0),median=float(np.median(runs)) if runs else 0.,p95=float(np.percentile(runs,95)) if runs else 0.)

def audit_plan(plan,c):
    mandatory_rows=[r for r in plan if r['kind']=='mandatory'];pairs=[plan[i:i+2] for i in range(0,len(plan),2)];composition=Counter(sum(r['positive'] for r in pair) for pair in pairs)
    rank_min=[]
    for rank in (0,1):
        flags=[r['positive'] for r in plan[rank::2]];w=c['rank_window_updates'];cs=np.r_[0,np.cumsum(flags)];rank_min.append(float((cs[w:]-cs[:-w]).min()/w) if len(flags)>=w else 0.)
    return dict(mandatory_samples=len(mandatory_rows),mandatory_unique=len({(r['exam'],tuple(r['origin']),r['z']) for r in mandatory_rows}),supplementary_samples=len(plan)-len(mandatory_rows),total_samples=len(plan),optimizer_updates=len(pairs),positive_samples=sum(r['positive'] for r in plan),positive_fraction=sum(r['positive'] for r in plan)/len(plan),negative_categories=dict(Counter(r['negative_category'] for r in plan if not r['positive'])),update_composition={'both_negative':composition[0],'mixed':composition[1],'both_positive':composition[2]},negative_only_streak=streaks([not any(r['positive'] for r in p) for p in pairs]),patient_repeat_streak=streaks([plan[i]['patient']==plan[i-1]['patient'] for i in range(1,len(plan))]),exam_repeat_streak=streaks([plan[i]['exam']==plan[i-1]['exam'] for i in range(1,len(plan))]),rank_window_min_positive_fraction=rank_min,mandatory_SHA=digest(mandatory_rows),plan_SHA=digest(plan),foreground_pixel_density='measured at loading, not inferred from positive sample category')

def commit_update(state,pair):
    if len(pair)!=2:raise ValueError('Frozen global batch two')
    for row in pair:
        state['mandatory_unique']+=int(row['kind']=='mandatory');state['replay_consumed']+=int(row['kind']=='replay')
        state['positive_samples']+=int(row['positive']);state['negative_samples']+=int(not row['positive'])
    state['cursor']+=2;state['step']+=1
    if state['cursor']!=2*state['step'] or state['mandatory_unique']+state['replay_consumed']!=state['cursor']:raise ValueError('Consumed-update accounting mismatch')
