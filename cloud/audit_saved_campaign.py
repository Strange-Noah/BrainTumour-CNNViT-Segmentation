"""Read-only, aggregate audit of the archived v1 campaign. R2 GET/HEAD only."""
import hashlib
import io
import json
from pathlib import Path

from cloud.storage import connect, verify

ACTIVE = 'novo-nordisk/experiments/unet25d/full-training-active.json'
COLLECTION = 'novo-nordisk/experiments/unet25d/bridge/44878438761/collection.json'
COLLECTION_SHA = '4fe1028e398817d1dd84597d120a9803ab31a5f7cac564275fe8cd6c2a804ccd'
ORIGINAL_SCIENCE = 'c3a7a581affc95bb5ea8723c9d286c7f95376717de40bbecdb03e2887bf427b0'
RECOVERED_SCIENCE = '14cf0e1591d6c120e5dbc072b7e90ca886f5010773f0c02ebcf57f5e351ad60c'


def read(client, bucket, key, expected=None):
    if expected is None:
        head = client.head_object(Bucket=bucket, Key=key)
        expected = head.get('Metadata', {}).get('sha256')
        if not expected:
            raise ValueError('Metadata has no SHA identity')
    dest = io.BytesIO()
    verify(client, bucket, key, expected, dest)
    return json.loads(dest.getvalue())


def validation_summary(data):
    rows = data['visits']
    if len(rows) != 159 or len({r['exam_id'] for r in rows}) != 159 or len({r['patient_id'] for r in rows}) != 45 or not data['complete']:
        raise ValueError('Incomplete or duplicate frozen validation')
    counts = {}
    for threshold in (.3, .5, .7):
        selected = rows if threshold == .5 else [r['threshold_grid'][str(threshold)] for r in rows]
        counts[str(threshold)] = dict(predicted_components=sum(r['matched_lesions'] + r['false_positives'] for r in selected), matched_lesions=sum(r['matched_lesions'] for r in selected), truth_lesions=sum(r['truth_lesions'] for r in selected))
    return dict(complete=True, patients=45, examinations=159, control=data['control'], secondary_threshold_grid=data['secondary_threshold_grid'], threshold_counts=counts, lesion_counts_at_0_5={s: dict(total=sum(r['lesion_strata'][s]['truth'] for r in rows), matched=sum(r['lesion_strata'][s]['matched'] for r in rows)) for s in ('lt100_mm3', '100_to_lt1000_mm3', 'ge1000_mm3')}, test_evaluated=data.get('test_evaluated'))


def main():
    client, bucket = connect(output=True)
    if bucket != 'novo-nordisk-pbl-outputs':
        raise ValueError('Unexpected output bucket')
    active = read(client, bucket, ACTIVE)
    collection = read(client, bucket, COLLECTION, COLLECTION_SHA)
    receipts = collection['receipts']
    if len(receipts) != 963 or len({r['key'] for r in receipts}) != 963 or not collection['independent_readback']:
        raise ValueError('Collection identity/count/readback mismatch')
    states = []; validations = []; checkpoints = []; histories = []
    for receipt in receipts:
        key = receipt['key']
        if key.endswith(('/checkpoint.pt', '/best-positive-dice.pt', '/best-tradeoff.pt')):
            head = client.head_object(Bucket=bucket, Key=key)
            if head['ContentLength'] != receipt['bytes'] or head.get('Metadata', {}).get('sha256') != receipt['sha256']:
                raise ValueError('Checkpoint receipt differs from durable object')
            checkpoints.append(dict(receipt, durable_head_verified=True, independent_GET_readback_basis='verified collection 44878438761'))
        elif key.endswith('/training-state.json'):
            states.append(read(client, bucket, key, receipt['sha256']))
        elif key.endswith('/validation.json'):
            validations.append(validation_summary(read(client, bucket, key, receipt['sha256'])))
        elif key.endswith('/adaptive-history.json'):
            control = read(client, bucket, key, receipt['sha256'])
            histories.append({k: control[k] for k in ('ratios', 'target_streak', 'stagnant', 'phase2_validations', 'last_adjustment', 'scientific_stop')})
    if len(states) != 1 or len(validations) != 2 or not checkpoints:
        raise ValueError('Required final evidence absent')
    pointers = {}
    for identity in (ORIGINAL_SCIENCE, RECOVERED_SCIENCE):
        pointer = read(client, bucket, 'novo-nordisk/experiments/unet25d/full-runs/' + identity + '/latest.json')
        pointers[identity] = pointer  # Pointer contains aggregate state and receipts, never patient records.
    # Never print the full active record: it can carry private dataset/package locators.
    safe_active = {k: active.get(k) for k in ('status', 'source_sha', 'science_sha256', 'session_no', 'kernel', 'approved_budget_seconds', 'reserved_seconds', 'submitted_at', 'session_seconds', 'initial_observation_complete')}
    report = dict(schema='saved-campaign-read-only-audit-v1', experiment_id='unet25d-v1', research_outcome='FAILED_BASELINE', active_record=safe_active, active_record_is_authoritative_for_provider_status=False, collection=dict(key=COLLECTION, sha256=COLLECTION_SHA, receipts=len(receipts), independent_readback=True), final_state=states[0], checkpoints=checkpoints, pointers=pointers, validations=sorted(validations, key=lambda x: x['control']['positive_patient_dice']), adaptive_controls=histories, operations=['GET', 'HEAD'], GPU_submitted=False, private_objects_modified=False)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
