import json, glob, sys, os
os.chdir('/home/johnson/workspace/LLM_PostProcess')
E='experiments/2026-08-17-teacher-b-corpus-review/results/'
errs=[]
def load(p):
    rows=[]
    for i,l in enumerate(open(p),1):
        if not l.strip(): errs.append(f"{p}:{i} blank line"); continue
        try: rows.append(json.loads(l))
        except Exception as e: errs.append(f"{p}:{i} parse {e}")
    return rows

corp={}
for split,f in (('train','research/ai-infra-expert/corpus/train.jsonl'),('validation','research/ai-infra-expert/corpus/validation.jsonl')):
    corp[split]=[json.loads(l) for l in open(f)]

REQ=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user","source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]
allids=[]
for split in ('train','validation'):
    files=sorted(glob.glob(E+split+'-batch-*.jsonl'))
    agg=[]
    for p in files:
        r=load(p); agg+=r
        if len(r)!=10 and p!=files[-1]: errs.append(f"{p} has {len(r)} rows")
    for idx,rec in enumerate(agg):
        for k in REQ:
            if k not in rec: errs.append(f"{split}[{idx}] missing {k}")
        if rec.get('teacher_lane')!='teacher-B': errs.append(f"{split}[{idx}] lane")
        if rec.get('teacher_model')!='claude-opus-5-current': errs.append(f"{split}[{idx}] model")
        if rec.get('calibration_status')!='provisional': errs.append(f"{split}[{idx}] status")
        if rec.get('decision') not in ('keep','rewrite','reject'): errs.append(f"{split}[{idx}] decision")
        if not isinstance(rec.get('corrected_answer'),str) or not rec['corrected_answer'].strip(): errs.append(f"{split}[{idx}] empty corrected_answer")
        c=rec.get('confidence')
        if not isinstance(c,(int,float)) or not (0<=c<=1): errs.append(f"{split}[{idx}] confidence")
        qd=rec.get('quality_dimensions',{})
        for k in ('technical_correctness','instruction_coverage','operational_safety'):
            v=qd.get(k)
            if not isinstance(v,int) or not(1<=v<=5): errs.append(f"{split}[{idx}] qd {k}")
        for k in ('risks','evidence_required'):
            if not isinstance(rec.get(k),list) or not all(isinstance(x,str) for x in rec[k]): errs.append(f"{split}[{idx}] {k}")
        # prefix check
        src=corp[split][idx]
        u=[x for x in src['messages'] if x['role']=='user'][0]['content']
        a=[x for x in src['messages'] if x['role']=='assistant'][0]['content']
        if rec.get('source_id')!=src['id']: errs.append(f"{split}[{idx}] id mismatch {rec.get('source_id')} vs {src['id']}")
        if rec.get('source_user')!=u: errs.append(f"{split}[{idx}] source_user mismatch")
        if rec.get('source_assistant')!=a: errs.append(f"{split}[{idx}] source_assistant mismatch")
        allids.append(rec.get('source_id'))
    print(split, len(agg), '/', len(corp[split]))
if len(set(allids))!=len(allids): errs.append("duplicate source_id")
print("ERRORS:", len(errs))
for e in errs[:30]: print(" ", e)
sys.exit(1 if errs else 0)
