import json, os, glob, hashlib, sys

RES='experiments/2026-08-17-teacher-b-corpus-review/results'
REQ={"source_id","teacher_lane","teacher_model","calibration_status","decision",
     "source_user","source_assistant","corrected_answer","quality_dimensions",
     "risks","evidence_required","confidence"}
fail=[]
def ck(c,m):
    if not c: fail.append(m)

def load_corpus(name):
    out=[]
    for l in open(f'research/ai-infra-expert/corpus/{name}.jsonl',encoding='utf-8'):
        r=json.loads(l); m=r['messages']
        out.append((r['id'],
                    [x for x in m if x['role']=='user'][0]['content'],
                    [x for x in m if x['role']=='assistant'][0]['content']))
    return out

# raw file checks on this batch
BF=f'{RES}/train-batch-0160.jsonl'
raw=open(BF,'rb').read()
ck(b'\r' not in raw,'CR present')
ck(raw.endswith(b'\n'),'no trailing newline')
lines=raw.decode('utf-8').splitlines()
ck(len(lines)==10,f'batch line count {len(lines)}')
batch=[json.loads(l) for l in lines]

hs=set()
for o in batch:
    ck(set(o.keys())==REQ, f'field mismatch {set(o.keys())^REQ}')
    ck(o['teacher_lane']=='teacher-B','lane')
    ck(o['teacher_model']=='claude-opus-5-current','model')
    ck(o['calibration_status']=='provisional','status')
    ck(o['decision'] in ('keep','rewrite','reject'),'decision')
    ck(isinstance(o['corrected_answer'],str) and len(o['corrected_answer'].strip())>0,'empty corrected')
    ck(o['corrected_answer']!=o['source_assistant'],'corrected==source')
    ck(isinstance(o['confidence'],float) and 0.0<=o['confidence']<=1.0,'confidence range')
    qd=o['quality_dimensions']
    ck(set(qd.keys())=={'technical_correctness','instruction_coverage','operational_safety'},'qd keys')
    for k,v in qd.items(): ck(isinstance(v,int) and 1<=v<=5,f'qd {k}')
    ck(isinstance(o['risks'],list) and o['risks'] and all(isinstance(x,str) and x for x in o['risks']),'risks')
    ck(isinstance(o['evidence_required'],list) and o['evidence_required'] and all(isinstance(x,str) and x for x in o['evidence_required']),'evidence')
    hs.add(hashlib.sha256(o['corrected_answer'].encode()).hexdigest())
ck(len(hs)==10,f'anti-template: only {len(hs)} distinct corrected_answer')

# global aggregate: prefix + uniqueness + exact source equality
for split,total in (('train',5399),('validation',601)):
    files=sorted(glob.glob(f'{RES}/{split}-batch-*.jsonl'))
    recs=[]
    for f_ in files:
        recs+= [json.loads(l) for l in open(f_,encoding='utf-8')]
    if not recs: continue
    corpus=load_corpus(split)
    ck(len(recs)<=total,f'{split} overflow')
    for i,o in enumerate(recs):
        cid,cu,ca=corpus[i]
        ck(o['source_id']==cid,f'{split} prefix break at {i}: {o["source_id"]}!={cid}')
        ck(o['source_user']==cu,f'{split} user mismatch at {i}')
        ck(o['source_assistant']==ca,f'{split} assistant mismatch at {i}')
    ids=[o['source_id'] for o in recs]
    ck(len(set(ids))==len(ids),f'{split} dup ids')
    print(f'{split}: {len(recs)}/{total}')

print('FAIL' if fail else 'PASS')
for f_ in fail[:20]: print(' -',f_)
sys.exit(1 if fail else 0)
