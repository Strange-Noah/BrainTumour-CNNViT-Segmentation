"""Training-only bounded difficulty memory and complete-validation research control."""
import copy, math
from collections import defaultdict

CATEGORIES = ('baseline', 'hard_positive', 'hard_negative', 'ordinary_negative')

def commit_update(state,plan,world=2):
    """Call only after a successful synchronized optimizer update, never prefetch."""
    cursor=state['global_cursor']
    if cursor%world or cursor+world>len(plan):raise ValueError('Invalid actual consumption')
    if state['phase']==1 and state['phase1_unique']!=cursor:raise ValueError('Exhaustive prefix mismatch')
    for d in plan[cursor:cursor+world]:state['category_consumed'][d[3]]=state['category_consumed'].get(d[3],0)+1
    state['global_cursor']+=world;state['step']+=1
    if state['phase']==1:state['phase1_unique']+=world;state['phase1_updates']+=1
    else:state['phase2_consumed']+=world

def validation_completed(state):
    if not state['pending_validation']:raise ValueError('No scheduled validation')
    state['pending_validation']=False;state['validation_progress']=[]
    if state['phase']==1:state['phase']=2
    else:state['phase2_cycle']+=1
    state['active_plan']=None;state['global_cursor']=0

class Difficulty:
    def __init__(self, visits, records=None):
        if any(v['partition'] != 'train' for v in visits):
            raise ValueError('Difficulty memory is training-only')
        self.allowed = {v['metadata']['exam_id'] for v in visits}
        self.records = records or {}
        if any(r['exam'] not in self.allowed for r in self.records.values()):
            raise ValueError('Foreign difficulty record')

    def update(self, rows, step):
        for r in sorted(rows, key=lambda r: (r['step'], r['rank'], r['exam'], r['z'])):
            if r['exam'] not in self.allowed: raise ValueError('Nontraining mining')
            key = str((r['exam'], r['tile'], r['z'], r['kind']))
            old = self.records.get(key)
            r = dict(r)
            r['score'] = .8 * old['score'] + .2 * r['score'] if old else r['score']
            if not math.isfinite(r['score']): raise ValueError('Nonfinite difficulty')
            self.records[key] = r
        groups = defaultdict(list)
        for k, r in self.records.items():
            if step - r['step'] <= 200000:
                groups[r['exam'], r['kind']].append((k, r))
        self.records = {}
        for rows in groups.values():
            rows.sort(key=lambda x: (-self.weight(x[1], step), x[0]))
            self.records.update(rows[:32])

    @staticmethod
    def weight(r, step):
        return max(1e-6, r['score'] * 2 ** (-(step-r['step'])/50000))

def initial_control():
    return dict(ratios=dict(zip(CATEGORIES, [.4,.25,.2,.15])), history=[], target_streak=0,
                stagnant=0, phase2_validations=0, last_adjustment=None, scientific_stop=None)

def feedback(control, metrics, *, phase2, exhaustive_complete):
    """Never update on partial validation; ratio changes do not reset patience."""
    if not metrics['complete']: raise ValueError('Partial validation cannot control training')
    if not exhaustive_complete: raise ValueError('Exhaustive phase not complete')
    c = copy.deepcopy(control); h = c['history']; previous = h[-1]['metrics'] if h else None
    dice, sens, fp = (metrics[k] for k in ('positive_patient_dice','patient_lesion_sensitivity','patient_fp_per_scan'))
    if any(v is None or not math.isfinite(v) for v in (dice,sens,fp)): raise ValueError('Undefined control metrics')
    reached = dice >= .70 and sens >= .80 and fp <= 2
    c['target_streak'] = c['target_streak']+1 if reached else 0
    if phase2: c['phase2_validations'] += 1
    improved = False; regressed = False
    if previous:
        dd=dice-previous['positive_patient_dice']; ds=sens-previous['patient_lesion_sensitivity']; df=previous['patient_fp_per_scan']-fp
        regressed = dd < -.01 or ds < -.02 or df < -.5
        improved = (dd >= .01 or ds >= .02 or df >= .5) and not regressed
        c['stagnant'] = 0 if improved else c['stagnant']+1
    reason = 'hold: assess unchanged mixture'; old = dict(c['ratios']); change = None
    both_worse = previous and sens < previous['patient_lesion_sensitivity']-.02 and fp > previous['patient_fp_per_scan']+.5
    loss_worse_quality = previous and metrics['validation_loss'] < previous['validation_loss'] and regressed
    last = c['last_adjustment']
    if last and (both_worse or loss_worse_quality):
        c['ratios'] = dict(last['before']); c['last_adjustment']=None; reason='revert unsuccessful last adjustment'
    elif improved:
        reason='hold: meaningful nonregressive improvement'; c['last_adjustment']=None
    elif last:
        # One complete unchanged cycle must assess a change before another bias.
        c['last_adjustment']=None; reason='hold: reassess previous adjustment'
    else:
        small=metrics['small_lesion_sensitivity'].get('lt100_mm3')
        if small is not None and small < .8 and fp <= 2: change=('hard_positive','ordinary_negative'); reason='small lesion sensitivity below target with acceptable FP'
        elif fp > 2 and (not previous or sens >= previous['patient_lesion_sensitivity']-.02): change=('hard_negative','ordinary_negative'); reason='excess FP with stable sensitivity; fixed threshold grid reported'
        elif sens >= .8 and fp <= 2 and dice < .7: change=('hard_positive','hard_negative'); reason='emphasize positive boundaries'
        if change:
            up,down=change
            if c['ratios'][down] <= .05: down = 'hard_negative' if up=='hard_positive' else 'hard_positive'
            delta=min(.10,max(0,c['ratios'][down]-.05))
            c['ratios'][up]+=delta; c['ratios'][down]-=delta
            if delta:c['last_adjustment']=dict(before=old,direction=up)
    if c['ratios']['baseline'] < .30-1e-9: raise ValueError('Baseline floor')
    if c['target_streak'] >= 2: c['scientific_stop']='TARGET_REACHED'
    elif c['phase2_validations'] >= 3 and c['stagnant'] >= 3: c['scientific_stop']='PLATEAU_BELOW_TARGET'
    h.append(dict(metrics=metrics,ratios_used=old,ratios_next=dict(c['ratios']),reason=reason,phase2=phase2))
    c['history']=h[-64:]
    return c
