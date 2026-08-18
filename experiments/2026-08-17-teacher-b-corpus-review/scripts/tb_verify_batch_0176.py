import json, hashlib, os, re, glob

EXP = '/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review'
CORPUS = '/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl'
REQUIRED = {'source_id','teacher_lane','teacher_model','calibration_status','decision',
            'source_user','source_assistant','corrected_answer','quality_dimensions',
            'risks','evidence_required','confidence'}
fails = []
def ck(c, m):
    if not c: fails.append(m)

# zero validation files
ck(glob.glob(os.path.join(EXP,'results','validation-batch-*')) == [], 'validation-batch files exist')

files = sorted(glob.glob(os.path.join(EXP,'results','train-batch-*.jsonl')))
nums = [int(re.search(r'(\d{4})\.jsonl$', f).group(1)) for f in files]
ck(nums == list(range(1, len(files)+1)), 'batch numbering not contiguous from 0001')

recs = []
for f in files:
    raw = open(f, encoding='utf-8').read()
    ck(raw.split('\n')[-1] == '', 'missing trailing newline: '+f)
    for ln in raw.split('\n'):
        if ln.strip():
            recs.append((f, json.loads(ln)))

B = [r for f, r in recs if f.endswith('0176.jsonl')]
ck(len(B) == 10, 'batch size != 10: %d' % len(B))

clines = open(CORPUS, encoding='utf-8').read().split('\n')
corpus = [json.loads(l) for l in clines if l.strip()]

for r in B:
    sid = r['source_id']
    ck(set(r.keys()) == REQUIRED, 'field set mismatch '+sid)
    ck(r['teacher_lane'] == 'teacher-B', 'lane '+sid)
    ck(r['teacher_model'] == 'claude-opus-5-current', 'model '+sid)
    ck(r['calibration_status'] == 'provisional', 'status '+sid)
    ck(r['decision'] in ('keep','rewrite','reject'), 'decision '+sid)
    ck(isinstance(r['corrected_answer'], str) and r['corrected_answer'].strip() != '', 'empty ca '+sid)
    ck(r['corrected_answer'] != r['source_assistant'], 'ca == source_assistant '+sid)
    ck('ESTIMATE' in r['corrected_answer'], 'missing ESTIMATE labelling '+sid)
    qd = r['quality_dimensions']
    ck(set(qd.keys()) == {'technical_correctness','instruction_coverage','operational_safety'}, 'qd keys '+sid)
    ck(all(isinstance(v,int) and 1<=v<=5 for v in qd.values()), 'qd range '+sid)
    ck(isinstance(r['risks'], list) and all(isinstance(x,str) for x in r['risks']), 'risks '+sid)
    ck(isinstance(r['evidence_required'], list) and all(isinstance(x,str) for x in r['evidence_required']), 'evreq '+sid)
    ck(isinstance(r['confidence'], float) and 0.0 <= r['confidence'] <= 1.0, 'confidence '+sid)

# anti-template
hs = {hashlib.sha256(r['corrected_answer'].encode()).hexdigest() for r in B}
ck(len(hs) == 10, 'duplicate corrected_answer hashes: %d' % len(hs))
mechs = []
for r in B:
    m = re.search(r'Analytical stance under test: (.*?)\.\n', r['corrected_answer'])
    ck(m is not None, 'missing stance marker '+r['source_id'])
    if m: mechs.append(m.group(1))
ck(len(set(mechs)) == len(mechs), 'duplicate stances in batch')
ck(len({r['corrected_answer'][:200] for r in B}) == 10, 'shared 200-char opening')

# global uniqueness + prefix
ids = [r['source_id'] for _, r in recs]
ck(len(ids) == len(set(ids)), 'duplicate source_id globally')
ck(ids == [c['id'] for c in corpus[:len(ids)]], 'aggregate is not a strict corpus prefix')

# byte-exact source fields against corpus
cmap = {c['id']: {m['role']: m['content'] for m in c['messages']} for c in corpus}
for _, r in recs:
    src = cmap[r['source_id']]
    ck(r['source_user'] == src['user'], 'source_user mismatch '+r['source_id'])
    ck(r['source_assistant'] == src['assistant'], 'source_assistant mismatch '+r['source_id'])

print('total train records:', len(recs))
print('batch 0176:', B[0]['source_id'], '->', B[-1]['source_id'])
print('decisions:', {d: sum(1 for r in B if r['decision']==d) for d in ('keep','rewrite','reject')})
print('RESULT:', 'PASS' if not fails else 'FAIL')
for f in fails: print('  FAIL:', f)
