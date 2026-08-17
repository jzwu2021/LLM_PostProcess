import json, glob, os, sys

BASE = 'experiments/2026-08-17-teacher-b-corpus-review/results'
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
errs = []

def load_corpus(p):
    out = []
    for l in open(p):
        d = json.loads(l)
        u = [m for m in d['messages'] if m['role']=='user'][0]['content']
        a = [m for m in d['messages'] if m['role']=='assistant'][0]['content']
        out.append((d['id'], u, a))
    return out

train_c = load_corpus('research/ai-infra-expert/corpus/train.jsonl')
val_c = load_corpus('research/ai-infra-expert/corpus/validation.jsonl')
CORPUS_LEN = {'train': len(train_c), 'validation': len(val_c)}

def collect(prefix):
    recs = []
    files = sorted(glob.glob(f'{BASE}/{prefix}-batch-*.jsonl'))
    for fi, fp in enumerate(files):
        n = 0
        for i, line in enumerate(open(fp), 1):
            line = line.rstrip('\n')
            if not line.strip():
                errs.append(f'{fp}:{i} empty line'); continue
            try:
                recs.append((fp, i, json.loads(line))); n += 1
            except Exception as e:
                errs.append(f'{fp}:{i} JSON parse fail: {e}')
        # every batch holds 10 records; only a batch that exhausts the corpus may be short
        if n != 10 and not (fi == len(files) - 1 and len(recs) == CORPUS_LEN[prefix]):
            errs.append(f'{fp} has {n} records, expected 10')
    return recs

seen = set()
for prefix, corpus in (('train', train_c), ('validation', val_c)):
    recs = collect(prefix)
    if len(recs) > len(corpus):
        errs.append(f'{prefix}: {len(recs)} records exceed corpus {len(corpus)}')
    for idx, (fp, ln, r) in enumerate(recs):
        tag = f'{fp}:{ln}'
        for k in REQ:
            if k not in r: errs.append(f'{tag} missing field {k}')
        if len(r) != 12: errs.append(f'{tag} has {len(r)} fields, expected 12')
        if r.get('teacher_lane') != 'teacher-B': errs.append(f'{tag} bad teacher_lane')
        if r.get('teacher_model') != 'claude-opus-5-current': errs.append(f'{tag} bad teacher_model')
        if r.get('calibration_status') != 'provisional': errs.append(f'{tag} bad calibration_status')
        if r.get('decision') not in ('keep','rewrite','reject'): errs.append(f'{tag} bad decision')
        ca = r.get('corrected_answer')
        if not isinstance(ca, str) or not ca.strip(): errs.append(f'{tag} empty corrected_answer')
        qd = r.get('quality_dimensions')
        if not isinstance(qd, dict): errs.append(f'{tag} quality_dimensions not object')
        else:
            for k in ('technical_correctness','instruction_coverage','operational_safety'):
                v = qd.get(k)
                if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= 5):
                    errs.append(f'{tag} bad quality_dimensions.{k}={v!r}')
            if len(qd) != 3: errs.append(f'{tag} quality_dimensions has {len(qd)} keys')
        for k in ('risks','evidence_required'):
            v = r.get(k)
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                errs.append(f'{tag} {k} not list[str]')
        c = r.get('confidence')
        if not isinstance(c, (int, float)) or isinstance(c, bool) or not (0.0 <= c <= 1.0):
            errs.append(f'{tag} confidence out of range: {c!r}')
        sid = r.get('source_id')
        if sid in seen: errs.append(f'{tag} duplicate source_id {sid}')
        seen.add(sid)
        if idx < len(corpus):
            cid, cu, ca_src = corpus[idx]
            if sid != cid: errs.append(f'{tag} prefix order break: got {sid}, corpus[{idx}]={cid}')
            if r.get('source_user') != cu: errs.append(f'{tag} source_user mismatch')
            if r.get('source_assistant') != ca_src: errs.append(f'{tag} source_assistant mismatch')
    print(f'{prefix}: {len(recs)}/{len(corpus)}')

print('ERRORS:', len(errs))
for e in errs[:40]: print(' ', e)
print('VERIFY_RESULT=' + ('PASS' if not errs else 'FAIL'))
sys.exit(1 if errs else 0)
