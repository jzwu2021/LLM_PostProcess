import json, glob, sys, os
EXP="experiments/2026-08-17-teacher-b-corpus-review/results"
errs=[]
def load_corpus(p):
    out=[]
    with open(p) as f:
        for l in f:
            r=json.loads(l)
            u=next(x["content"] for x in r["messages"] if x["role"]=="user")
            a=next(x["content"] for x in r["messages"] if x["role"]=="assistant")
            out.append((r["id"],u,a))
    return out
tr=load_corpus("research/ai-infra-expert/corpus/train.jsonl")
va=load_corpus("research/ai-infra-expert/corpus/validation.jsonl")
REQ=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user","source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]

def collect(prefix):
    seq=[]
    for f in sorted(glob.glob(f"{EXP}/{prefix}-batch-*.jsonl")):
        with open(f) as fh:
            for ln,l in enumerate(fh,1):
                if not l.strip(): continue
                try: r=json.loads(l)
                except Exception as e: errs.append(f"{f}:{ln} parse {e}"); continue
                seq.append((f,ln,r))
    return seq

allids=[]
for prefix,corpus in (("train",tr),("validation",va)):
    seq=collect(prefix)
    for i,(f,ln,r) in enumerate(seq):
        for k in REQ:
            if k not in r: errs.append(f"{f}:{ln} missing {k}")
        if r.get("teacher_lane")!="teacher-B": errs.append(f"{f}:{ln} lane")
        if r.get("teacher_model")!="claude-opus-5-current": errs.append(f"{f}:{ln} model")
        if r.get("calibration_status")!="provisional": errs.append(f"{f}:{ln} status")
        if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{f}:{ln} decision")
        if not isinstance(r.get("corrected_answer"),str) or not r["corrected_answer"].strip(): errs.append(f"{f}:{ln} empty answer")
        c=r.get("confidence")
        if not isinstance(c,(int,float)) or not 0<=c<=1: errs.append(f"{f}:{ln} confidence")
        qd=r.get("quality_dimensions",{})
        for d in ("technical_correctness","instruction_coverage","operational_safety"):
            v=qd.get(d)
            if not isinstance(v,int) or not 1<=v<=5: errs.append(f"{f}:{ln} qd {d}")
        for d in ("risks","evidence_required"):
            if not isinstance(r.get(d),list) or not all(isinstance(x,str) for x in r.get(d,[])): errs.append(f"{f}:{ln} {d}")
        if i>=len(corpus): errs.append(f"{f}:{ln} beyond corpus"); continue
        cid,cu,ca=corpus[i]
        if r.get("source_id")!=cid: errs.append(f"{f}:{ln} prefix-order expected {cid} got {r.get('source_id')}")
        if r.get("source_user")!=cu: errs.append(f"{f}:{ln} source_user mismatch")
        if r.get("source_assistant")!=ca: errs.append(f"{f}:{ln} source_assistant mismatch")
        allids.append(r.get("source_id"))
    print(prefix,"count",len(seq))
if len(allids)!=len(set(allids)): errs.append("duplicate source_id globally")
print("ERRORS",len(errs))
for e in errs[:30]: print(e)
sys.exit(1 if errs else 0)
