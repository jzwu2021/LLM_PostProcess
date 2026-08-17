import json, glob, sys, os
base='experiments/2026-08-17-teacher-b-corpus-review/results'
fail=[]
REQ=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user","source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]

def load_corpus(p):
    out=[]
    for l in open(p):
        d=json.loads(l)
        m={x['role']:x['content'] for x in d['messages']}
        out.append((d['id'], m['user'], m['assistant']))
    return out

train=load_corpus('research/ai-infra-expert/corpus/train.jsonl')
val=load_corpus('research/ai-infra-expert/corpus/validation.jsonl')

def agg(prefix):
    files=sorted(glob.glob(f'{base}/{prefix}-batch-*.jsonl'))
    recs=[]
    for f in files:
        for i,l in enumerate(open(f),1):
            l=l.rstrip('\n')
            if not l: continue
            try: recs.append((f,i,json.loads(l)))
            except Exception as e: fail.append(f'{f}:{i} JSON parse: {e}')
    return files,recs

tf,tr=agg('train'); vf,vr=agg('validation')
allr=tr+vr
seen={}
for f,i,r in allr:
    for k in REQ:
        if k not in r: fail.append(f'{f}:{i} missing field {k}')
    if r.get('teacher_lane')!='teacher-B': fail.append(f'{f}:{i} bad teacher_lane')
    if r.get('teacher_model')!='claude-opus-5-current': fail.append(f'{f}:{i} bad teacher_model')
    if r.get('calibration_status')!='provisional': fail.append(f'{f}:{i} bad calibration_status')
    if r.get('decision') not in ('keep','rewrite','reject'): fail.append(f'{f}:{i} bad decision')
    if not isinstance(r.get('corrected_answer'),str) or not r['corrected_answer'].strip(): fail.append(f'{f}:{i} empty corrected_answer')
    c=r.get('confidence')
    if not isinstance(c,(int,float)) or not (0.0<=c<=1.0): fail.append(f'{f}:{i} bad confidence')
    qd=r.get('quality_dimensions')
    if not isinstance(qd,dict): fail.append(f'{f}:{i} qd not object')
    else:
        for k in ("technical_correctness","instruction_coverage","operational_safety"):
            v=qd.get(k)
            if not isinstance(v,int) or isinstance(v,bool) or not (1<=v<=5): fail.append(f'{f}:{i} qd.{k} bad')
    for k in ("risks","evidence_required"):
        v=r.get(k)
        if not isinstance(v,list) or not all(isinstance(x,str) for x in v): fail.append(f'{f}:{i} {k} not str[]')
    sid=r.get('source_id')
    if sid in seen: fail.append(f'{f}:{i} duplicate source_id {sid} (also {seen[sid]})')
    else: seen[sid]=f'{f}:{i}'

def prefix_check(recs, corpus, name):
    if len(recs)>len(corpus): fail.append(f'{name}: more records ({len(recs)}) than corpus ({len(corpus)})'); return
    for idx,(f,i,r) in enumerate(recs):
        cid,cu,ca=corpus[idx]
        if r.get('source_id')!=cid: fail.append(f'{name}[{idx}] {f}:{i} id {r.get("source_id")} != corpus {cid}')
        if r.get('source_user')!=cu: fail.append(f'{name}[{idx}] {f}:{i} source_user mismatch')
        if r.get('source_assistant')!=ca: fail.append(f'{name}[{idx}] {f}:{i} source_assistant mismatch')

prefix_check(tr,train,'train'); prefix_check(vr,val,'validation')

newb=sys.argv[1] if len(sys.argv)>1 else None
if newb:
    n=sum(1 for l in open(f'{base}/{newb}') if l.strip())
    if n!=10: fail.append(f'{newb} has {n} records, expected 10')

print(f'train={len(tr)}/{len(train)} validation={len(vr)}/{len(val)} total={len(tr)+len(vr)}/6000')
if fail:
    print('VERIFY_FAIL', len(fail))
    for x in fail[:40]: print(' ', x)
    sys.exit(1)
print('VERIFY_PASS')
