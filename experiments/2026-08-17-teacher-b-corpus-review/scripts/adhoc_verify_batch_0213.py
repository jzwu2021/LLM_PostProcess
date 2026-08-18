import json, glob
BASE = '/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review'
files = sorted(glob.glob(BASE + '/results/train-batch-*.jsonl'))
rows = []
for f in files:
    for l in open(f):
        if l.strip():
            rows.append(json.loads(l))
corpus = [json.loads(l) for l in open('/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl') if l.strip()]
assert len(rows) == 2130, len(rows)
ids = [r['source_id'] for r in rows]
assert len(set(ids)) == len(ids), 'duplicate source_id'
assert ids == [c['id'] for c in corpus[:2130]], 'not a strict prefix of train.jsonl'
b = [json.loads(l) for l in open(BASE + '/results/train-batch-0213.jsonl')]
assert len(b) == 10
req = ["source_id", "teacher_lane", "teacher_model", "calibration_status", "decision",
       "source_user", "source_assistant", "corrected_answer", "quality_dimensions",
       "risks", "evidence_required", "confidence"]
for r, c in zip(b, corpus[2120:2130]):
    m = {x['role']: x['content'] for x in c['messages']}
    assert set(r) == set(req), set(r) ^ set(req)
    assert r['teacher_lane'] == 'teacher-B'
    assert r['teacher_model'] == 'claude-opus-5-current'
    assert r['calibration_status'] == 'provisional'
    assert r['decision'] in ('keep', 'rewrite', 'reject')
    assert r['source_user'] == m['user']
    assert r['source_assistant'] == m['assistant']
    assert isinstance(r['corrected_answer'], str) and r['corrected_answer'].strip()
    assert isinstance(r['confidence'], float) and 0.0 <= r['confidence'] <= 1.0
    assert isinstance(r['risks'], list) and all(isinstance(x, str) for x in r['risks'])
    assert isinstance(r['evidence_required'], list) and all(isinstance(x, str) for x in r['evidence_required'])
    qd = r['quality_dimensions']
    assert set(qd) == {"technical_correctness", "instruction_coverage", "operational_safety"}
    assert all(isinstance(qd[k], int) and 1 <= qd[k] <= 5 for k in qd)
print('ADHOC_PASS total', len(rows))
