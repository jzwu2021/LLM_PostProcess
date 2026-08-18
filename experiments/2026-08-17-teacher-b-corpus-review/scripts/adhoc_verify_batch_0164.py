import json, os, glob, sys, hashlib

ROOT = '/home/johnson/workspace/LLM_PostProcess'
EXP = os.path.join(ROOT, 'experiments/2026-08-17-teacher-b-corpus-review')
CORPUS = os.path.join(ROOT, 'research/ai-infra-expert/corpus/train.jsonl')
BATCH = os.path.join(EXP, 'results/train-batch-0164.jsonl')
EXPECT_N = 10
FIELDS = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
          "source_user","source_assistant","corrected_answer","quality_dimensions",
          "risks","evidence_required","confidence"]
fails = []
def chk(c, m):
    if not c: fails.append(m)

raw = open(BATCH, 'rb').read().decode('utf-8')
lines = raw.split('\n')
chk(lines[-1] == '', 'batch file must end with newline')
lines = [l for l in lines if l != '']
chk(len(lines) == EXPECT_N, f'expected {EXPECT_N} lines, got {len(lines)}')

recs = []
for i, l in enumerate(lines):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fails.append(f'line {i+1} not valid JSON: {e}')

corpus = [json.loads(l) for l in open(CORPUS)]
cmap = {}
for row in corpus:
    m = row['messages']
    cmap[row['id']] = ([x for x in m if x['role']=='user'][0]['content'],
                       [x for x in m if x['role']=='assistant'][0]['content'])

for i, r in enumerate(recs):
    tag = f'rec{i+1}({r.get("source_id")})'
    for f in FIELDS:
        chk(f in r, f'{tag}: missing field {f}')
    chk(len(r.keys()) == 12, f'{tag}: expected exactly 12 fields, got {len(r.keys())}')
    chk(r.get('teacher_lane') == 'teacher-B', f'{tag}: bad teacher_lane')
    chk(r.get('teacher_model') == 'claude-opus-5-current', f'{tag}: bad teacher_model')
    chk(r.get('calibration_status') == 'provisional', f'{tag}: bad calibration_status')
    chk(r.get('decision') in ('keep','rewrite','reject'), f'{tag}: bad decision')
    ca = r.get('corrected_answer')
    chk(isinstance(ca, str) and len(ca.strip()) > 0, f'{tag}: empty corrected_answer')
    qd = r.get('quality_dimensions')
    chk(isinstance(qd, dict), f'{tag}: quality_dimensions not object')
    if isinstance(qd, dict):
        chk(set(qd.keys()) == {'technical_correctness','instruction_coverage','operational_safety'},
            f'{tag}: bad quality_dimensions keys')
        for k, v in qd.items():
            chk(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5, f'{tag}: {k}={v} not int 1-5')
    chk(isinstance(r.get('risks'), list) and all(isinstance(x, str) for x in r['risks']), f'{tag}: risks not str[]')
    chk(isinstance(r.get('evidence_required'), list) and all(isinstance(x, str) for x in r['evidence_required']), f'{tag}: evidence_required not str[]')
    c = r.get('confidence')
    chk(isinstance(c, float) and 0.0 <= c <= 1.0, f'{tag}: confidence out of range')
    sid = r.get('source_id')
    chk(sid in cmap, f'{tag}: source_id not in corpus')
    if sid in cmap:
        u, a = cmap[sid]
        chk(r.get('source_user') == u, f'{tag}: source_user mismatch')
        chk(r.get('source_assistant') == a, f'{tag}: source_assistant mismatch')

# global uniqueness + strict prefix of train.jsonl
allf = sorted(glob.glob(os.path.join(EXP, 'results/train-batch-*.jsonl')))
agg = []
for p in allf:
    for l in open(p):
        l = l.strip()
        if l: agg.append(json.loads(l)['source_id'])
chk(len(agg) == len(set(agg)), f'duplicate source_id across batches ({len(agg)-len(set(agg))} dups)')
corpus_ids = [r['id'] for r in corpus]
chk(agg == corpus_ids[:len(agg)], 'aggregate train sequence is NOT a strict prefix of train.jsonl')

# no validation batches must exist
chk(len(glob.glob(os.path.join(EXP, 'results/validation-batch-*.jsonl'))) == 0, 'validation batch files present')

# anti-template
hs = [hashlib.sha256(r['corrected_answer'].encode()).hexdigest() for r in recs]
chk(len(set(hs)) == len(hs), 'duplicate corrected_answer within batch')

print('batch lines:', len(recs), '| aggregate train:', len(agg), '| batch files:', len(allf))
if fails:
    print('VERIFY_FAIL')
    for f in fails: print(' -', f)
    sys.exit(1)
print('VERIFY_PASS')
