import json, os, glob, re, sys
os.chdir("/home/johnson/workspace/LLM_PostProcess")
D = "experiments/2026-08-17-teacher-b-corpus-review/results"
errs = []

def load_corpus(p):
    out = []
    for line in open(p):
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        u = [m for m in d["messages"] if m["role"] == "user"][0]["content"]
        a = [m for m in d["messages"] if m["role"] == "assistant"][0]["content"]
        out.append((d["id"], u, a))
    return out

corpora = {"train": load_corpus("research/ai-infra-expert/corpus/train.jsonl"),
           "validation": load_corpus("research/ai-infra-expert/corpus/validation.jsonl")}
print("corpus sizes:", {k: len(v) for k, v in corpora.items()})

REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user",
       "source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]

seen = set()
for split in ["train","validation"]:
    files = sorted(glob.glob(f"{D}/{split}-batch-*.jsonl"))
    agg = []
    for fp in files:
        raw = open(fp).read()
        lines = [l for l in raw.split("\n") if l.strip()]
        if raw and not raw.endswith("\n"): errs.append(f"{fp}: no trailing newline")
        if len(lines) != 10 and fp == files[-1]:
            pass
        for i, l in enumerate(lines):
            try: r = json.loads(l)
            except Exception as e: errs.append(f"{fp}:{i+1} parse {e}"); continue
            for k in REQ:
                if k not in r: errs.append(f"{fp}:{i+1} missing {k}")
            if r.get("teacher_lane") != "teacher-B": errs.append(f"{fp}:{i+1} lane")
            if r.get("teacher_model") != "claude-opus-5-current": errs.append(f"{fp}:{i+1} model")
            if r.get("calibration_status") != "provisional": errs.append(f"{fp}:{i+1} status")
            if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{fp}:{i+1} decision")
            if not isinstance(r.get("corrected_answer"),str) or not r["corrected_answer"].strip():
                errs.append(f"{fp}:{i+1} empty corrected_answer")
            c = r.get("confidence")
            if not isinstance(c,(int,float)) or not (0.0 <= c <= 1.0): errs.append(f"{fp}:{i+1} confidence")
            qd = r.get("quality_dimensions")
            if not isinstance(qd,dict): errs.append(f"{fp}:{i+1} qd type")
            else:
                for k in ["technical_correctness","instruction_coverage","operational_safety"]:
                    v = qd.get(k)
                    if not isinstance(v,int) or not (1<=v<=5): errs.append(f"{fp}:{i+1} qd.{k}")
            if not isinstance(r.get("risks"),list) or not all(isinstance(x,str) for x in r["risks"]): errs.append(f"{fp}:{i+1} risks")
            if not isinstance(r.get("evidence_required"),list) or not all(isinstance(x,str) for x in r["evidence_required"]): errs.append(f"{fp}:{i+1} evidence_required")
            sid = r.get("source_id")
            if sid in seen: errs.append(f"{fp}:{i+1} duplicate source_id {sid}")
            seen.add(sid)
            agg.append(r)
    # prefix check
    corp = corpora[split]
    if len(agg) > len(corp): errs.append(f"{split}: more records than corpus")
    for i, r in enumerate(agg):
        cid, cu, ca = corp[i]
        if r["source_id"] != cid: errs.append(f"{split}[{i}] id mismatch {r['source_id']} != {cid}"); break
        if r["source_user"] != cu: errs.append(f"{split}[{i}] source_user mismatch")
        if r["source_assistant"] != ca: errs.append(f"{split}[{i}] source_assistant mismatch")
    print(f"{split}: {len(agg)} records over {len(files)} files")

if errs:
    print("FAIL", len(errs)); [print(" ", e) for e in errs[:40]]; sys.exit(1)
print("VERIFY_PASS")
