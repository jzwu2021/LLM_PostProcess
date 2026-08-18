import json, sys, os, glob, re

EXP="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
RES=os.path.join(EXP,"results")
CORP="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus"
BATCH=os.path.join(RES,"train-batch-0158.jsonl")
fails=[]
def ck(c,m):
    if not c: fails.append(m)

REQ=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user",
     "source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]

def load_corpus(name):
    out=[]
    with open(os.path.join(CORP,name)) as f:
        for l in f:
            d=json.loads(l)
            u=[m["content"] for m in d["messages"] if m["role"]=="user"][0]
            a=[m["content"] for m in d["messages"] if m["role"]=="assistant"][0]
            out.append((d["id"],u,a))
    return out
train=load_corpus("train.jsonl"); val=load_corpus("validation.jsonl")
tmap={i:(u,a) for i,u,a in train}; vmap={i:(u,a) for i,u,a in val}

# 1. this batch
raw=open(BATCH,"rb").read()
ck(raw.endswith(b"\n"),"batch not newline terminated")
lines=raw.decode().split("\n")[:-1]
ck(len(lines)==10,f"batch line count {len(lines)} != 10")
recs=[]
for n,l in enumerate(lines,1):
    try: r=json.loads(l)
    except Exception as e: fails.append(f"line {n} unparseable: {e}"); continue
    recs.append(r)
    for k in REQ: ck(k in r, f"line {n} missing {k}")
    ck(r.get("teacher_lane")=="teacher-B", f"line {n} bad lane")
    ck(r.get("teacher_model")=="claude-opus-5-current", f"line {n} bad model")
    ck(r.get("calibration_status")=="provisional", f"line {n} bad status")
    ck(r.get("decision") in ("keep","rewrite","reject"), f"line {n} bad decision")
    ck(isinstance(r.get("corrected_answer"),str) and r["corrected_answer"].strip(), f"line {n} empty corrected_answer")
    c=r.get("confidence"); ck(isinstance(c,(int,float)) and 0<=c<=1, f"line {n} bad confidence")
    qd=r.get("quality_dimensions")
    ck(isinstance(qd,dict), f"line {n} qd not obj")
    if isinstance(qd,dict):
        for d in ("technical_correctness","instruction_coverage","operational_safety"):
            v=qd.get(d); ck(isinstance(v,int) and not isinstance(v,bool) and 1<=v<=5, f"line {n} qd.{d} bad: {v}")
    ck(isinstance(r.get("risks"),list) and all(isinstance(x,str) for x in r["risks"]), f"line {n} risks bad")
    ck(isinstance(r.get("evidence_required"),list) and all(isinstance(x,str) for x in r["evidence_required"]), f"line {n} evidence bad")
    sid=r.get("source_id")
    src=tmap.get(sid)
    ck(src is not None, f"line {n} source_id {sid} not in train corpus")
    if src:
        ck(r.get("source_user")==src[0], f"line {n} source_user mismatch")
        ck(r.get("source_assistant")==src[1], f"line {n} source_assistant mismatch")

# 2. global aggregation: prefix + uniqueness
def agg(prefix):
    ids=[]
    for p in sorted(glob.glob(os.path.join(RES,prefix+"-batch-*.jsonl"))):
        with open(p) as f:
            for l in f:
                if l.strip(): ids.append(json.loads(l)["source_id"])
    return ids
tids=agg("train"); vids=agg("validation")
ck(len(set(tids))==len(tids), "duplicate source_id in train aggregate")
ck(len(set(vids))==len(vids), "duplicate source_id in validation aggregate")
ck(not (set(tids)&set(vids)), "train/validation source_id overlap")
ck(tids==[i for i,_,_ in train][:len(tids)], "train aggregate is not a strict corpus prefix")
ck(vids==[i for i,_,_ in val][:len(vids)], "validation aggregate is not a strict corpus prefix")

print(f"train_total={len(tids)} validation_total={len(vids)} total={len(tids)+len(vids)}")
if fails:
    print("VERIFY_FAIL"); [print(" -",f) for f in fails[:40]]; sys.exit(1)
print("VERIFY_PASS")
