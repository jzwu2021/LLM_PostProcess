import json, glob, os, re, sys
ROOT="/home/johnson/workspace/LLM_PostProcess"
CORPUS=f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP=f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
BATCH=f"{EXP}/results/train-batch-0196.jsonl"
errs=[]
def ck(c,msg):
    if not c: errs.append(msg)

corpus=[json.loads(l) for l in open(CORPUS) if l.strip()]
cmap={r["id"]:r for r in corpus}
corder=[r["id"] for r in corpus]

lines=open(BATCH).read().split("\n")
lines=[l for l in lines if l.strip()]
ck(len(lines)==10, f"batch count {len(lines)} != 10")
rows=[]
for i,l in enumerate(lines):
    try: rows.append(json.loads(l))
    except Exception as e: errs.append(f"line {i+1} parse: {e}")

REQ=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user",
     "source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]
for r in rows:
    sid=r.get("source_id")
    for k in REQ: ck(k in r, f"{sid} missing {k}")
    ck(r.get("teacher_lane")=="teacher-B", f"{sid} lane")
    ck(r.get("teacher_model")=="claude-opus-5-current", f"{sid} model")
    ck(r.get("calibration_status")=="provisional", f"{sid} status")
    ck(r.get("decision") in ("keep","rewrite","reject"), f"{sid} decision")
    src=cmap.get(sid)
    ck(src is not None, f"{sid} not in corpus")
    if src:
        m={x['role']:x['content'] for x in src['messages']}
        ck(r["source_user"]==m["user"], f"{sid} source_user mismatch")
        ck(r["source_assistant"]==m["assistant"], f"{sid} source_assistant mismatch")
    ca=r.get("corrected_answer","")
    ck(isinstance(ca,str) and len(ca.strip())>0, f"{sid} empty corrected_answer")
    ck(ca!=r.get("source_assistant"), f"{sid} corrected==source_assistant")
    ck("ESTIMATE" in ca, f"{sid} no ESTIMATE label")
    ck(ca.startswith("Analytical stance under test:"), f"{sid} missing stance marker")
    qd=r.get("quality_dimensions",{})
    ck(isinstance(qd,dict) and set(qd)=={"technical_correctness","instruction_coverage","operational_safety"}, f"{sid} qd keys")
    for k,v in (qd or {}).items():
        ck(isinstance(v,int) and 1<=v<=5, f"{sid} qd {k}={v}")
    for k in ("risks","evidence_required"):
        v=r.get(k)
        ck(isinstance(v,list) and len(v)>0 and all(isinstance(x,str) and x.strip() for x in v), f"{sid} {k} bad")
    c=r.get("confidence")
    ck(isinstance(c,(int,float)) and 0.0<=c<=1.0, f"{sid} confidence {c}")

ck(len({r["corrected_answer"][:200] for r in rows})==10, "opening 200 chars not distinct")
ck(len({r["corrected_answer"] for r in rows})==10, "answers not distinct")

# global aggregate
files=sorted(glob.glob(f"{EXP}/results/train-batch-*.jsonl"))
allids=[]
for f in files:
    for l in open(f):
        if l.strip(): allids.append(json.loads(l)["source_id"])
ck(len(allids)==len(set(allids)), "duplicate source_id across batches")
ck(allids==corder[:len(allids)], "aggregated sequence is not a strict prefix of train.jsonl")
print("TOTAL", len(allids))
vb=glob.glob(f"{EXP}/results/validation-batch-*.jsonl")
ck(len(vb)==0, f"validation batch files exist: {vb}")

if errs:
    print("FAIL", len(errs))
    for e in errs[:40]: print(" -", e)
    sys.exit(1)
print("VERIFY_PASS")
