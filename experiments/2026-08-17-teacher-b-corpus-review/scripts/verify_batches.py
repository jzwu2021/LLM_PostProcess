import json, glob, os, sys

BASE = 'experiments/2026-08-17-teacher-b-corpus-review/results'
errs = []

def load_corpus(p):
    recs = []
    for l in open(p):
        d = json.loads(l)
        su = [m['content'] for m in d['messages'] if m['role'] == 'user'][0]
        sa = [m['content'] for m in d['messages'] if m['role'] == 'assistant'][0]
        recs.append((d['id'], su, sa))
    return recs

train = load_corpus('research/ai-infra-expert/corpus/train.jsonl')
val = load_corpus('research/ai-infra-expert/corpus/validation.jsonl')

REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]

def collect(prefix):
    out = []
    for fp in sorted(glob.glob(f'{BASE}/{prefix}-batch-*.jsonl')):
        raw = open(fp).read()
        lines = raw.split('\n')
        if lines and lines[-1] == '':
            lines.pop()
        for i, l in enumerate(lines):
            try:
                out.append((fp, i, json.loads(l)))
            except Exception as e:
                errs.append(f'{fp}:{i} parse error {e}')
    return out

tr = collect('train')
va = collect('validation')

newbatch = f'{BASE}/train-batch-0055.jsonl'
n_new = sum(1 for fp, i, r in tr if fp == newbatch)
if n_new != 10:
    errs.append(f'new batch count {n_new} != 10')

seen = set()
for fp, i, r in tr + va:
    for k in REQ:
        if k not in r:
            errs.append(f'{fp}:{i} missing {k}')
    if r.get('teacher_lane') != 'teacher-B': errs.append(f'{fp}:{i} bad lane')
    if r.get('teacher_model') != 'claude-opus-5-current': errs.append(f'{fp}:{i} bad model')
    if r.get('calibration_status') != 'provisional': errs.append(f'{fp}:{i} bad status')
    if r.get('decision') not in ('keep','rewrite','reject'): errs.append(f'{fp}:{i} bad decision')
    if not isinstance(r.get('corrected_answer'), str) or not r['corrected_answer'].strip():
        errs.append(f'{fp}:{i} empty corrected_answer')
    c = r.get('confidence')
    if not isinstance(c,(int,float)) or not (0.0 <= c <= 1.0): errs.append(f'{fp}:{i} bad confidence')
    qd = r.get('quality_dimensions')
    if not isinstance(qd, dict): errs.append(f'{fp}:{i} qd not object')
    else:
        for k in ('technical_correctness','instruction_coverage','operational_safety'):
            v = qd.get(k)
            if not isinstance(v,int) or not (1 <= v <= 5): errs.append(f'{fp}:{i} qd {k}')
    for k in ('risks','evidence_required'):
        v = r.get(k)
        if not isinstance(v,list) or not all(isinstance(x,str) for x in v): errs.append(f'{fp}:{i} {k} not str list')
    sid = r.get('source_id')
    if sid in seen: errs.append(f'{fp}:{i} duplicate source_id {sid}')
    seen.add(sid)

def check_prefix(rows, corpus, name):
    if len(rows) > len(corpus):
        errs.append(f'{name} longer than corpus')
        return
    for idx, (fp, i, r) in enumerate(rows):
        cid, cu, ca = corpus[idx]
        if r.get('source_id') != cid: errs.append(f'{name} idx {idx} id mismatch {r.get("source_id")} != {cid}')
        if r.get('source_user') != cu: errs.append(f'{name} idx {idx} source_user mismatch')
        if r.get('source_assistant') != ca: errs.append(f'{name} idx {idx} source_assistant mismatch')

check_prefix(tr, train, 'train')
check_prefix(va, val, 'validation')

print(f'train={len(tr)}/5399 validation={len(va)}/601 total={len(tr)+len(va)}/6000')
if errs:
    print('VERIFY_FAIL')
    for e in errs[:40]: print(e)
    sys.exit(1)
print('VERIFY_PASS')
