#!/usr/bin/env python3
import json, os, glob, sys
ROOT="/home/johnson/workspace/LLM_PostProcess"
CORPUS=os.path.join(ROOT,"research/ai-infra-expert/corpus/train.jsonl")
RESDIR=os.path.join(ROOT,"experiments/2026-08-17-teacher-b-corpus-review/results")
BATCH=os.path.join(RESDIR,"train-batch-0172.jsonl")
REQ=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user","source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]
errs=[]
corpus=[]
with open(CORPUS,encoding="utf-8") as f:
    for l in f:
        r=json.loads(l)
        m=r["messages"]
        corpus.append((r["id"],
                       next(x["content"] for x in m if x["role"]=="user"),
                       next(x["content"] for x in m if x["role"]=="assistant")))
cmap={c[0]:c for c in corpus}

recs=[]
with open(BATCH,encoding="utf-8") as f:
    for i,l in enumerate(f,1):
        try: recs.append(json.loads(l))
        except Exception as e: errs.append(f"line {i} parse: {e}")
if len(recs)!=10: errs.append(f"batch count {len(recs)}!=10")
for r in recs:
    for k in REQ:
        if k not in r: errs.append(f"{r.get('source_id')} missing {k}")
    if r.get("teacher_lane")!="teacher-B": errs.append("lane")
    if r.get("teacher_model")!="claude-opus-5-current": errs.append("model")
    if r.get("calibration_status")!="provisional": errs.append("status")
    if r.get("decision") not in ("keep","rewrite","reject"): errs.append("decision")
    if not isinstance(r.get("corrected_answer"),str) or not r["corrected_answer"].strip(): errs.append("empty answer")
    c=r.get("confidence")
    if not isinstance(c,(int,float)) or not 0<=c<=1: errs.append("confidence")
    qd=r.get("quality_dimensions",{})
    for d in ("technical_correctness","instruction_coverage","operational_safety"):
        if not isinstance(qd.get(d),int) or not 1<=qd[d]<=5: errs.append(f"qd {d}")
    if not isinstance(r.get("risks"),list) or not all(isinstance(x,str) for x in r["risks"]): errs.append("risks")
    if not isinstance(r.get("evidence_required"),list) or not all(isinstance(x,str) for x in r["evidence_required"]): errs.append("evidence")
    src=cmap.get(r["source_id"])
    if not src: errs.append(f"unknown id {r['source_id']}")
    else:
        if r["source_user"]!=src[1]: errs.append(f"{r['source_id']} user mismatch")
        if r["source_assistant"]!=src[2]: errs.append(f"{r['source_id']} assistant mismatch")

# global aggregate
allrecs=[]
for p in sorted(glob.glob(os.path.join(RESDIR,"train-batch-*.jsonl"))):
    with open(p,encoding="utf-8") as f:
        for l in f: allrecs.append(json.loads(l))
ids=[r["source_id"] for r in allrecs]
if len(ids)!=len(set(ids)): errs.append("duplicate source_id globally")
for i,(rid) in enumerate(ids):
    if rid!=corpus[i][0]: errs.append(f"prefix break at {i}: {rid} != {corpus[i][0]}"); break
if glob.glob(os.path.join(RESDIR,"*validation*")): errs.append("validation file present")
print("total_train =",len(allrecs))
print("ERRORS:",errs if errs else "NONE")
sys.exit(1 if errs else 0)
