import json, glob, os, sys, re

ROOT = "experiments/2026-08-17-teacher-b-corpus-review/results"
CORP = {"train": "research/ai-infra-expert/corpus/train.jsonl",
        "validation": "research/ai-infra-expert/corpus/validation.jsonl"}
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
errs = []

corpus = {}
for k,p in CORP.items():
    rows=[]
    for ln in open(p,encoding="utf-8"):
        d=json.loads(ln)
        m=d["messages"]
        rows.append((d["id"],
                     [x for x in m if x["role"]=="user"][0]["content"],
                     [x for x in m if x["role"]=="assistant"][0]["content"]))
    corpus[k]=rows

seen=set()
seq={"train":[], "validation":[]}
for split in ("train","validation"):
    files = sorted(glob.glob(f"{ROOT}/{split}-batch-*.jsonl"))
    for fp in files:
        raw = open(fp,encoding="utf-8").read()
        lines = raw.split("\n")
        if lines and lines[-1]=="": lines.pop()
        if len(lines)!=10 and fp!=files[-1]:
            errs.append(f"{fp}: expected 10 lines, got {len(lines)}")
        for i,ln in enumerate(lines,1):
            try: r=json.loads(ln)
            except Exception as e:
                errs.append(f"{fp}:{i} JSON parse: {e}"); continue
            for f in REQ:
                if f not in r: errs.append(f"{fp}:{i} missing field {f}")
            if r.get("teacher_lane")!="teacher-B": errs.append(f"{fp}:{i} bad teacher_lane")
            if r.get("teacher_model")!="claude-opus-5-current": errs.append(f"{fp}:{i} bad teacher_model")
            if r.get("calibration_status")!="provisional": errs.append(f"{fp}:{i} bad calibration_status")
            if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{fp}:{i} bad decision")
            ca=r.get("corrected_answer")
            if not isinstance(ca,str) or not ca.strip(): errs.append(f"{fp}:{i} empty corrected_answer")
            c=r.get("confidence")
            if not isinstance(c,(int,float)) or not (0.0<=c<=1.0): errs.append(f"{fp}:{i} bad confidence")
            qd=r.get("quality_dimensions")
            if not isinstance(qd,dict): errs.append(f"{fp}:{i} quality_dimensions not object")
            else:
                for d3 in ("technical_correctness","instruction_coverage","operational_safety"):
                    v=qd.get(d3)
                    if not isinstance(v,int) or isinstance(v,bool) or not (1<=v<=5):
                        errs.append(f"{fp}:{i} bad quality_dimensions.{d3}")
            for lf in ("risks","evidence_required"):
                v=r.get(lf)
                if not isinstance(v,list) or not all(isinstance(x,str) for x in v):
                    errs.append(f"{fp}:{i} {lf} not string array")
            sid=r.get("source_id")
            if sid in seen: errs.append(f"{fp}:{i} duplicate source_id {sid}")
            seen.add(sid)
            seq[split].append((sid, r.get("source_user"), r.get("source_assistant")))

for split in ("train","validation"):
    got=seq[split]; exp=corpus[split][:len(got)]
    if len(got)>len(corpus[split]):
        errs.append(f"{split}: more records ({len(got)}) than corpus ({len(corpus[split])})")
    for i,(g,e) in enumerate(zip(got,exp)):
        if g[0]!=e[0]: errs.append(f"{split}[{i}] id mismatch {g[0]} != {e[0]}")
        if g[1]!=e[1]: errs.append(f"{split}[{i}] source_user mismatch for {g[0]}")
        if g[2]!=e[2]: errs.append(f"{split}[{i}] source_assistant mismatch for {g[0]}")

print(f"train={len(seq['train'])}/{len(corpus['train'])} validation={len(seq['validation'])}/{len(corpus['validation'])} total={len(seq['train'])+len(seq['validation'])}")
if errs:
    print("VERIFY_FAIL", len(errs))
    for e in errs[:40]: print("  ", e)
    sys.exit(1)
print("VERIFY_PASS")
