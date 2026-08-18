import json, os, glob, sys

EXP = '/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review'
CORPUS = '/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl'
BATCH = os.path.join(EXP, 'results', 'train-batch-0179.jsonl')
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
fails = []

corpus = [json.loads(l) for l in open(CORPUS, encoding='utf-8')]
cmap = {}
for c in corpus:
    m = {x['role']: x['content'] for x in c['messages']}
    cmap[c['id']] = (m['user'], m['assistant'])

raw = open(BATCH, encoding='utf-8').read()
lines = [l for l in raw.split('\n') if l.strip()]
if len(lines) != 10:
    fails.append('batch line count %d != 10' % len(lines))
recs = []
for i, l in enumerate(lines):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fails.append('line %d parse error %s' % (i, e))

for r in recs:
    sid = r.get('source_id')
    for k in REQ:
        if k not in r:
            fails.append('%s missing %s' % (sid, k))
    if r.get('teacher_lane') != 'teacher-B': fails.append('%s lane' % sid)
    if r.get('teacher_model') != 'claude-opus-5-current': fails.append('%s model' % sid)
    if r.get('calibration_status') != 'provisional': fails.append('%s status' % sid)
    if r.get('decision') not in ('keep','rewrite','reject'): fails.append('%s decision' % sid)
    if sid not in cmap:
        fails.append('%s not in corpus' % sid); continue
    u, a = cmap[sid]
    if r['source_user'] != u: fails.append('%s user mismatch' % sid)
    if r['source_assistant'] != a: fails.append('%s assistant mismatch' % sid)
    if not isinstance(r.get('corrected_answer'), str) or not r['corrected_answer'].strip():
        fails.append('%s empty corrected_answer' % sid)
    qd = r.get('quality_dimensions')
    if not isinstance(qd, dict): fails.append('%s qd type' % sid)
    else:
        for k in ('technical_correctness','instruction_coverage','operational_safety'):
            v = qd.get(k)
            if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= 5):
                fails.append('%s qd %s' % (sid, k))
    if not isinstance(r.get('risks'), list) or not all(isinstance(x,str) for x in r['risks']):
        fails.append('%s risks' % sid)
    if not isinstance(r.get('evidence_required'), list) or not all(isinstance(x,str) for x in r['evidence_required']):
        fails.append('%s evidence' % sid)
    cf = r.get('confidence')
    if not isinstance(cf, (int,float)) or isinstance(cf, bool) or not (0.0 <= cf <= 1.0):
        fails.append('%s confidence' % sid)

# global aggregate: uniqueness + strict prefix of corpus order
allids = []
for f in sorted(glob.glob(os.path.join(EXP, 'results', 'train-batch-*.jsonl'))):
    for l in open(f, encoding='utf-8'):
        if l.strip():
            allids.append(json.loads(l)['source_id'])
if len(allids) != len(set(allids)):
    from collections import Counter
    dup = [k for k,v in Counter(allids).items() if v>1]
    fails.append('duplicate source_ids: %s' % dup[:10])
corpus_ids = [c['id'] for c in corpus]
if allids != corpus_ids[:len(allids)]:
    for i,(x,y) in enumerate(zip(allids, corpus_ids)):
        if x != y:
            fails.append('prefix break at %d: %s vs %s' % (i,x,y)); break
    else:
        fails.append('prefix length mismatch')

print('aggregate_train=%d' % len(allids))
print('this_batch=%d' % len(recs))
print('FAILS=%d' % len(fails))
for f in fails[:30]: print(' -', f)
sys.exit(1 if fails else 0)
