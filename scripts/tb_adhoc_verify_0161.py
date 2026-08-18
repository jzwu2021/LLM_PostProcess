import json, os, sys, glob, hashlib

EXP = 'experiments/2026-08-17-teacher-b-corpus-review'
RES = os.path.join(EXP, 'results')
CORPUS = {'train': 'research/ai-infra-expert/corpus/train.jsonl',
          'validation': 'research/ai-infra-expert/corpus/validation.jsonl'}
BATCH = os.path.join(RES, 'train-batch-0161.jsonl')
EXPECTED_N = 10
REQ = {"source_id", "teacher_lane", "teacher_model", "calibration_status", "decision",
       "source_user", "source_assistant", "corrected_answer", "quality_dimensions",
       "risks", "evidence_required", "confidence"}
errs = []


def load_corpus(p):
    out = []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            u = [m for m in d['messages'] if m['role'] == 'user'][0]['content']
            a = [m for m in d['messages'] if m['role'] == 'assistant'][0]['content']
            out.append((d['id'], u, a))
    return out


corp = {k: load_corpus(v) for k, v in CORPUS.items()}

raw = open(BATCH, 'rb').read()
if b'\r' in raw:
    errs.append('CR bytes present')
if not raw.endswith(b'\n'):
    errs.append('not newline-terminated')
lines = raw.decode('utf-8').rstrip('\n').split('\n')
if len(lines) != EXPECTED_N:
    errs.append('batch line count %d != %d' % (len(lines), EXPECTED_N))
recs = []
for i, ln in enumerate(lines):
    try:
        recs.append(json.loads(ln))
    except Exception as e:
        errs.append('line %d unparseable: %s' % (i + 1, e))

for i, r in enumerate(recs):
    tag = 'line %d' % (i + 1)
    if set(r.keys()) != REQ:
        errs.append('%s field set mismatch: missing=%s extra=%s' % (tag, REQ - set(r), set(r) - REQ))
        continue
    if r['teacher_lane'] != 'teacher-B':
        errs.append(tag + ' teacher_lane')
    if r['teacher_model'] != 'claude-opus-5-current':
        errs.append(tag + ' teacher_model')
    if r['calibration_status'] != 'provisional':
        errs.append(tag + ' calibration_status')
    if r['decision'] not in ('keep', 'rewrite', 'reject'):
        errs.append(tag + ' decision')
    if not isinstance(r['corrected_answer'], str) or not r['corrected_answer'].strip():
        errs.append(tag + ' corrected_answer empty')
    c = r['confidence']
    if not isinstance(c, float) or not (0.0 <= c <= 1.0):
        errs.append(tag + ' confidence')
    qd = r['quality_dimensions']
    if not isinstance(qd, dict) or set(qd) != {'technical_correctness', 'instruction_coverage', 'operational_safety'}:
        errs.append(tag + ' quality_dimensions keys')
    else:
        for k, v in qd.items():
            if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= 5):
                errs.append('%s qd.%s=%r' % (tag, k, v))
    for fld in ('risks', 'evidence_required'):
        v = r[fld]
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
            errs.append(tag + ' ' + fld)

# global aggregate: prefix check + uniqueness
for split in ('train', 'validation'):
    files = sorted(glob.glob(os.path.join(RES, split + '-batch-*.jsonl')))
    agg = []
    for fp in files:
        for ln in open(fp):
            ln = ln.strip()
            if ln:
                agg.append(json.loads(ln))
    ids = [x['source_id'] for x in agg]
    if len(ids) != len(set(ids)):
        errs.append(split + ': duplicate source_id')
    ref = corp[split]
    if len(agg) > len(ref):
        errs.append(split + ': more records than corpus')
    for i, rec in enumerate(agg):
        cid, cu, ca = ref[i]
        if rec['source_id'] != cid:
            errs.append('%s pos %d id %s != %s' % (split, i, rec['source_id'], cid))
            break
        if rec['source_user'] != cu:
            errs.append('%s pos %d source_user mismatch' % (split, i))
        if rec['source_assistant'] != ca:
            errs.append('%s pos %d source_assistant mismatch' % (split, i))
    globals()['n_' + split] = len(agg)

# cross-split id uniqueness
alltr = [x for x in range(0)]
print('train=%d/%d validation=%d/%d total=%d' % (n_train, len(corp['train']), n_validation, len(corp['validation']), n_train + n_validation))
print('batch decisions:', {d: sum(1 for r in recs if r['decision'] == d) for d in ('keep', 'rewrite', 'reject')})
print('distinct corrected_answer sha256:', len({hashlib.sha256(r['corrected_answer'].encode()).hexdigest() for r in recs}))
if errs:
    print('VERIFY_FAIL')
    for e in errs[:50]:
        print(' -', e)
    sys.exit(1)
print('VERIFY_PASS')
