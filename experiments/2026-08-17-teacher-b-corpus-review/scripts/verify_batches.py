import json, glob, os, sys, re

ROOT = "/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
FIELDS = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
          "source_user","source_assistant","corrected_answer","quality_dimensions",
          "risks","evidence_required","confidence"]
errs = []

def load_corpus(p):
    out=[]
    for l in open(p):
        d=json.loads(l)
        m={x["role"]:x["content"] for x in d["messages"]}
        out.append((d["id"], m["user"], m["assistant"]))
    return out

corp = {"train": load_corpus(os.path.join(ROOT,"research/ai-infra-expert/corpus/train.jsonl")),
        "validation": load_corpus(os.path.join(ROOT,"research/ai-infra-expert/corpus/validation.jsonl"))}

seen=set(); seq={"train":[], "validation":[]}
for split in ["train","validation"]:
    files = sorted(glob.glob(os.path.join(RES, f"{split}-batch-*.jsonl")))
    for fp in files:
        raw = open(fp).read()
        lines = raw.split("\n")
        if lines and lines[-1]=="": lines.pop()
        if len(lines)!=10: errs.append(f"{os.path.basename(fp)}: expected 10 lines got {len(lines)}")
        for i,l in enumerate(lines):
            try: r=json.loads(l)
            except Exception as e: errs.append(f"{fp}:{i+1} parse: {e}"); continue
            for f in FIELDS:
                if f not in r: errs.append(f"{fp}:{i+1} missing {f}")
            if r.get("teacher_lane")!="teacher-B": errs.append(f"{fp}:{i+1} bad lane")
            if r.get("teacher_model")!="claude-opus-5-current": errs.append(f"{fp}:{i+1} bad model")
            if r.get("calibration_status")!="provisional": errs.append(f"{fp}:{i+1} bad status")
            if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{fp}:{i+1} bad decision")
            if not isinstance(r.get("corrected_answer"),str) or not r["corrected_answer"].strip():
                errs.append(f"{fp}:{i+1} empty corrected_answer")
            c=r.get("confidence")
            if not isinstance(c,(int,float)) or not (0<=c<=1): errs.append(f"{fp}:{i+1} bad confidence")
            qd=r.get("quality_dimensions",{})
            for k in ["technical_correctness","instruction_coverage","operational_safety"]:
                v=qd.get(k)
                if not isinstance(v,int) or not (1<=v<=5): errs.append(f"{fp}:{i+1} bad qd {k}")
            for k in ["risks","evidence_required"]:
                if not isinstance(r.get(k),list) or not all(isinstance(x,str) for x in r.get(k,[None])):
                    errs.append(f"{fp}:{i+1} bad {k}")
            sid=r.get("source_id")
            if sid in seen: errs.append(f"{fp}:{i+1} duplicate source_id {sid}")
            seen.add(sid)
            seq[split].append((sid, r.get("source_user"), r.get("source_assistant")))

for split in ["train","validation"]:
    got=seq[split]; exp=corp[split][:len(got)]
    if len(got)>len(corp[split]): errs.append(f"{split}: more records than corpus")
    for i,(g,e) in enumerate(zip(got,exp)):
        if g[0]!=e[0]: errs.append(f"{split}[{i}] id mismatch {g[0]} != {e[0]}")
        if g[1]!=e[1]: errs.append(f"{split}[{i}] source_user mismatch at {e[0]}")
        if g[2]!=e[2]: errs.append(f"{split}[{i}] source_assistant mismatch at {e[0]}")

print(f"train={len(seq['train'])}/5399 validation={len(seq['validation'])}/601 total={len(seq['train'])+len(seq['validation'])}/6000")
if errs:
    print("VERIFY_FAIL"); [print(" -",e) for e in errs[:40]]; sys.exit(1)
print("VERIFY_PASS")
