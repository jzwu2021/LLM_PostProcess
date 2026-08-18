import json, glob, sys

BASE = 'experiments/2026-08-17-teacher-b-corpus-review'
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
errs = []

def load_corpus(p):
    out = []
    for l in open(p):
        r = json.loads(l)
        m = r['messages']
        out.append((r['id'],
                    [x['content'] for x in m if x['role']=='user'][0],
                    [x['content'] for x in m if x['role']=='assistant'][0]))
    return out

tr = load_corpus('research/ai-infra-expert/corpus/train.jsonl')
va = load_corpus('research/ai-infra-expert/corpus/validation.jsonl')
cmap = {i:(u,a) for i,u,a in tr+va}

seen = {}
seqs = {'train':[], 'validation':[]}
newest = BASE + '/results/train-batch-0127.jsonl'

for f in sorted(glob.glob(BASE+'/results/*.jsonl')):
    split = 'train' if '/train-' in f else 'validation'
    n = 0
    for ln, line in enumerate(open(f), 1):
        line = line.rstrip('\n')
        if not line: continue
        try:
            r = json.loads(line)
        except Exception as e:
            errs.append(f'{f}:{ln} JSON parse: {e}'); continue
        n += 1
        for k in REQ:
            if k not in r: errs.append(f'{f}:{ln} missing field {k}')
        if r.get('teacher_lane') != 'teacher-B': errs.append(f'{f}:{ln} bad lane')
        if r.get('teacher_model') != 'claude-opus-5-current': errs.append(f'{f}:{ln} bad model')
        if r.get('calibration_status') != 'provisional': errs.append(f'{f}:{ln} bad status')
        if r.get('decision') not in ('keep','rewrite','reject'): errs.append(f'{f}:{ln} bad decision')
        sid = r.get('source_id')
        if sid in seen: errs.append(f'{f}:{ln} duplicate source_id {sid} (also {seen[sid]})')
        seen[sid] = f
        if sid not in cmap:
            errs.append(f'{f}:{ln} unknown source_id {sid}')
        else:
            u,a = cmap[sid]
            if r.get('source_user') != u: errs.append(f'{f}:{ln} source_user mismatch {sid}')
            if r.get('source_assistant') != a: errs.append(f'{f}:{ln} source_assistant mismatch {sid}')
        if not isinstance(r.get('corrected_answer'), str) or not r.get('corrected_answer').strip():
            errs.append(f'{f}:{ln} empty corrected_answer')
        qd = r.get('quality_dimensions')
        if not isinstance(qd, dict): errs.append(f'{f}:{ln} qd not object')
        else:
            for k in ('technical_correctness','instruction_coverage','operational_safety'):
                v = qd.get(k)
                if not isinstance(v,int) or isinstance(v,bool) or not (1<=v<=5):
                    errs.append(f'{f}:{ln} qd.{k} invalid: {v}')
        if not isinstance(r.get('risks'), list) or not all(isinstance(x,str) for x in r.get('risks',[])):
            errs.append(f'{f}:{ln} risks invalid')
        if not isinstance(r.get('evidence_required'), list) or not all(isinstance(x,str) for x in r.get('evidence_required',[])):
            errs.append(f'{f}:{ln} evidence_required invalid')
        c = r.get('confidence')
        if not isinstance(c,(int,float)) or isinstance(c,bool) or not (0.0<=c<=1.0):
            errs.append(f'{f}:{ln} confidence invalid')
        seqs[split].append(sid)
    if f == newest and n != 10:
        errs.append(f'{f} expected 10 records got {n}')

for split, corp in (('train',tr), ('validation',va)):
    got = seqs[split]
    exp = [x[0] for x in corp][:len(got)]
    if got != exp:
        bad = [i for i,(g,e) in enumerate(zip(got,exp)) if g!=e][:5]
        errs.append(f'{split} not a strict prefix of corpus; first mismatches at {bad}')

print('train_processed', len(seqs['train']))
print('validation_processed', len(seqs['validation']))
print('total', len(seqs['train'])+len(seqs['validation']))
print('unique_ids', len(seen))
if errs:
    print('VERIFY_FAIL', len(errs))
    for e in errs[:30]: print(' -', e)
    sys.exit(1)
print('VERIFY_PASS')
