import json,glob,os,sys
ROOT="/home/johnson/workspace/LLM_PostProcess"
RES=f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review/results"
FIELDS=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user","source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]
errs=[]
def load_corpus(p):
    out=[]
    for l in open(p):
        d=json.loads(l); m={x["role"]:x["content"] for x in d["messages"]}
        out.append((d["id"],m["user"],m["assistant"]))
    return out
corp={"train":load_corpus(f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"),
      "validation":load_corpus(f"{ROOT}/research/ai-infra-expert/corpus/validation.jsonl")}
seen=set(); seq={"train":[],"validation":[]}
for split in ("train","validation"):
    files=sorted(glob.glob(f"{RES}/{split}-batch-*.jsonl"))
    for fp in files:
        raw=open(fp).read()
        lines=[x for x in raw.split("\n") if x.strip()]
        if not raw.endswith("\n"): errs.append(f"{fp}: no trailing newline")
        if len(lines)!=10 and fp==files[-1] and False: pass
        for i,l in enumerate(lines):
            try: r=json.loads(l)
            except Exception as e: errs.append(f"{fp}:{i+1} parse {e}"); continue
            for f in FIELDS:
                if f not in r: errs.append(f"{fp}:{i+1} missing {f}")
            if r.get("teacher_lane")!="teacher-B": errs.append(f"{fp}:{i+1} lane")
            if r.get("teacher_model")!="claude-opus-5-current": errs.append(f"{fp}:{i+1} model")
            if r.get("calibration_status")!="provisional": errs.append(f"{fp}:{i+1} status")
            if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{fp}:{i+1} decision")
            if not isinstance(r.get("corrected_answer"),str) or not r["corrected_answer"].strip(): errs.append(f"{fp}:{i+1} empty corrected_answer")
            c=r.get("confidence")
            if not isinstance(c,(int,float)) or not (0<=c<=1): errs.append(f"{fp}:{i+1} confidence")
            qd=r.get("quality_dimensions",{})
            for k in ("technical_correctness","instruction_coverage","operational_safety"):
                v=qd.get(k)
                if not isinstance(v,int) or not (1<=v<=5): errs.append(f"{fp}:{i+1} qd {k}")
            if not isinstance(r.get("risks"),list) or not all(isinstance(x,str) for x in r.get("risks",[])): errs.append(f"{fp}:{i+1} risks")
            if not isinstance(r.get("evidence_required"),list) or not all(isinstance(x,str) for x in r.get("evidence_required",[])): errs.append(f"{fp}:{i+1} evidence_required")
            sid=r.get("source_id")
            if sid in seen: errs.append(f"{fp}:{i+1} dup source_id {sid}")
            seen.add(sid)
            seq[split].append((sid,r.get("source_user"),r.get("source_assistant")))
for split in ("train","validation"):
    n=len(seq[split])
    pref=corp[split][:n]
    if len(pref)<n: errs.append(f"{split}: more records ({n}) than corpus ({len(corp[split])})")
    for i,(a,b) in enumerate(zip(seq[split],pref)):
        if a[0]!=b[0]: errs.append(f"{split}[{i}] id mismatch {a[0]} vs {b[0]}")
        elif a[1]!=b[1]: errs.append(f"{split}[{i}] source_user mismatch {a[0]}")
        elif a[2]!=b[2]: errs.append(f"{split}[{i}] source_assistant mismatch {a[0]}")
print("train_processed",len(seq["train"]),"validation_processed",len(seq["validation"]),"total",len(seq["train"])+len(seq["validation"]))
print("ERRORS",len(errs))
for e in errs[:30]: print(" ",e)
sys.exit(1 if errs else 0)
