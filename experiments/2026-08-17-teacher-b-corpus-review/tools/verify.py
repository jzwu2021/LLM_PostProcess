import json, glob, os, sys

EXP = 'experiments/2026-08-17-teacher-b-corpus-review/results'
fails = []

def load_corpus(p):
    out = []
    for l in open(p):
        r = json.loads(l)
        out.append((r['id'],
                    [m['content'] for m in r['messages'] if m['role'] == 'user'][0],
                    [m['content'] for m in r['messages'] if m['role'] == 'assistant'][0]))
    return out

train = load_corpus('research/ai-infra-expert/corpus/train.jsonl')
val = load_corpus('research/ai-infra-expert/corpus/validation.jsonl')

REQ = ["source_id", "teacher_lane", "teacher_model", "calibration_status", "decision",
       "source_user", "source_assistant", "corrected_answer", "quality_dimensions",
       "risks", "evidence_required", "confidence"]

def read_split(prefix):
    recs = []
    for f in sorted(glob.glob(f'{EXP}/{prefix}-batch-*.jsonl')):
        for i, line in enumerate(open(f), 1):
            line = line.rstrip('\n')
            if not line:
                continue
            try:
                recs.append((f, i, json.loads(line)))
            except Exception as e:
                fails.append(f'{f}:{i} JSON parse error: {e}')
    return recs

allids = []
for prefix, corpus in (('train', train), ('validation', val)):
    recs = read_split(prefix)
    for f, i, r in recs:
        for k in REQ:
            if k not in r:
                fails.append(f'{f}:{i} missing field {k}')
        if r.get('teacher_lane') != 'teacher-B':
            fails.append(f'{f}:{i} bad teacher_lane')
        if r.get('teacher_model') != 'claude-opus-5-current':
            fails.append(f'{f}:{i} bad teacher_model')
        if r.get('calibration_status') != 'provisional':
            fails.append(f'{f}:{i} bad calibration_status')
        if r.get('decision') not in ('keep', 'rewrite', 'reject'):
            fails.append(f'{f}:{i} bad decision')
        if not isinstance(r.get('corrected_answer'), str) or not r['corrected_answer'].strip():
            fails.append(f'{f}:{i} empty corrected_answer')
        qd = r.get('quality_dimensions')
        if not isinstance(qd, dict):
            fails.append(f'{f}:{i} quality_dimensions not object')
        else:
            for d in ('technical_correctness', 'instruction_coverage', 'operational_safety'):
                v = qd.get(d)
                if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 5:
                    fails.append(f'{f}:{i} bad {d}')
        for arr in ('risks', 'evidence_required'):
            v = r.get(arr)
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                fails.append(f'{f}:{i} {arr} not string array')
        c = r.get('confidence')
        if not isinstance(c, (int, float)) or isinstance(c, bool) or not 0.0 <= c <= 1.0:
            fails.append(f'{f}:{i} bad confidence')
        allids.append(r.get('source_id'))
    # prefix check
    for idx, (f, i, r) in enumerate(recs):
        if idx >= len(corpus):
            fails.append(f'{f}:{i} beyond corpus length')
            break
        cid, cu, ca = corpus[idx]
        if r.get('source_id') != cid:
            fails.append(f'{f}:{i} order/prefix mismatch: {r.get("source_id")} != {cid}')
        if r.get('source_user') != cu:
            fails.append(f'{f}:{i} source_user mismatch for {cid}')
        if r.get('source_assistant') != ca:
            fails.append(f'{f}:{i} source_assistant mismatch for {cid}')
    print(f'{prefix}: {len(recs)} records')

if len(allids) != len(set(allids)):
    seen, dup = set(), set()
    for x in allids:
        if x in seen:
            dup.add(x)
        seen.add(x)
    fails.append(f'duplicate source_ids: {sorted(dup)[:20]}')

print('total:', len(allids))
if fails:
    print('VERIFY_FAIL', len(fails))
    for x in fails[:50]:
        print(' -', x)
    sys.exit(1)
print('VERIFY_PASS')
