"""Independent ad-hoc verifier for teacher-B batch 0168 (re-reads corpus from disk)."""
import json, glob, os, sys, re

EXP = 'experiments/2026-08-17-teacher-b-corpus-review'
SRC = 'research/ai-infra-expert/corpus/train.jsonl'
BATCH = EXP + '/results/train-batch-0168.jsonl'
EXPECT_N = 10
REQUIRED = ["source_id", "teacher_lane", "teacher_model", "calibration_status",
            "decision", "source_user", "source_assistant", "corrected_answer",
            "quality_dimensions", "risks", "evidence_required", "confidence"]
fail = []


def chk(cond, msg):
    if not cond:
        fail.append(msg)


# ---- 1. raw JSONL parse, physical newline separated ----
raw = open(BATCH, 'rb').read().decode('utf-8')
lines = raw.split('\n')
chk(lines[-1] == '', 'batch file must end with a trailing newline')
lines = [l for l in lines if l != '']
chk(len(lines) == EXPECT_N, f'expected {EXPECT_N} lines, got {len(lines)}')
recs = []
for i, l in enumerate(lines):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fail.append(f'line {i+1} not valid JSON: {e}')

# ---- 2. corpus source of truth ----
corpus = [json.loads(l) for l in open(SRC)]
by_id = {}
for idx, c in enumerate(corpus):
    by_id[c['id']] = (idx, c)

for i, r in enumerate(recs):
    tag = f'rec{i+1}({r.get("source_id")})'
    for k in REQUIRED:
        chk(k in r, f'{tag}: missing field {k}')
    chk(len(r.keys()) == 12, f'{tag}: expected exactly 12 fields, got {len(r.keys())}')
    chk(r.get('teacher_lane') == 'teacher-B', f'{tag}: bad teacher_lane')
    chk(r.get('teacher_model') == 'claude-opus-5-current', f'{tag}: bad teacher_model')
    chk(r.get('calibration_status') == 'provisional', f'{tag}: bad calibration_status')
    chk(r.get('decision') in ('keep', 'rewrite', 'reject'), f'{tag}: bad decision')
    ca = r.get('corrected_answer')
    chk(isinstance(ca, str) and ca.strip() != '', f'{tag}: corrected_answer empty')
    conf = r.get('confidence')
    chk(isinstance(conf, float) and 0.0 <= conf <= 1.0, f'{tag}: confidence out of range')
    qd = r.get('quality_dimensions')
    chk(isinstance(qd, dict) and set(qd.keys()) == {
        'technical_correctness', 'instruction_coverage', 'operational_safety'},
        f'{tag}: quality_dimensions keys wrong')
    if isinstance(qd, dict):
        for k, v in qd.items():
            chk(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5,
                f'{tag}: quality_dimensions.{k} not int 1-5')
    for k in ('risks', 'evidence_required'):
        v = r.get(k)
        chk(isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) and x.strip() for x in v),
            f'{tag}: {k} must be a non-empty list of non-empty strings')
    # exact corpus equality
    sid = r.get('source_id')
    if sid not in by_id:
        fail.append(f'{tag}: source_id not in corpus')
    else:
        _, c = by_id[sid]
        cu = [m for m in c['messages'] if m['role'] == 'user'][0]['content']
        ca_src = [m for m in c['messages'] if m['role'] == 'assistant'][0]['content']
        chk(r.get('source_user') == cu, f'{tag}: source_user differs from corpus')
        chk(r.get('source_assistant') == ca_src, f'{tag}: source_assistant differs from corpus')
    # ESTIMATE/MEASURED tagging when digits with units appear
    if re.search(r'\b\d+(\.\d+)?\s*(GB/s|Gbps|GbE|ms|s|percent|%)\b', ca or ''):
        chk(('ESTIMATE' in ca) or ('MEASURED' in ca), f'{tag}: numeric claim without ESTIMATE/MEASURED tag')

# ---- 3. global uniqueness + strict prefix of corpus ----
all_ids = []
files = sorted(glob.glob(EXP + '/results/train-batch-*.jsonl'))
for f in files:
    for l in open(f):
        all_ids.append(json.loads(l)['source_id'])
chk(len(all_ids) == len(set(all_ids)), 'duplicate source_id across aggregated train batches')
corpus_ids = [c['id'] for c in corpus]
chk(all_ids == corpus_ids[:len(all_ids)],
    'aggregated train sequence is NOT a strict prefix of train.jsonl order')

# ---- 4. no validation artifacts in this phase ----
vfiles = glob.glob(EXP + '/results/*validation*')
chk(not vfiles, f'validation files must not exist: {vfiles}')

# ---- 5. anti-template: corrected_answer distinct within batch ----
cas = [r.get('corrected_answer') for r in recs]
chk(len(set(cas)) == len(cas), 'duplicate corrected_answer within batch 0168')

print(f'aggregated train records = {len(all_ids)}')
print(f'batch 0168 ids: {recs[0]["source_id"]} -> {recs[-1]["source_id"]}')
if fail:
    print('VERIFY_FAIL')
    for m in fail:
        print(' -', m)
    sys.exit(1)
print('VERIFY_PASS')
