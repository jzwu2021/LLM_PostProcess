import json, glob, os, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
RES = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review/results"
BATCH = f"{RES}/train-batch-0145.jsonl"
FIELDS = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
          "source_user","source_assistant","corrected_answer","quality_dimensions",
          "risks","evidence_required","confidence"]
errs = []

def corpus(name):
    out = []
    with open(f"{ROOT}/research/ai-infra-expert/corpus/{name}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            m = d["messages"]
            out.append((d["id"],
                        [x for x in m if x["role"]=="user"][0]["content"],
                        [x for x in m if x["role"]=="assistant"][0]["content"]))
    return out

TR, VA = corpus("train"), corpus("validation")
tr_map = {c[0]: c for c in TR}
va_map = {c[0]: c for c in VA}

# --- batch-level checks ---
raw = open(BATCH, "rb").read()
if not raw.endswith(b"\n"): errs.append("batch does not end with newline")
lines = raw.decode().rstrip("\n").split("\n")
if len(lines) != 10: errs.append(f"batch line count {len(lines)} != 10")
recs = []
for i, ln in enumerate(lines):
    try: recs.append(json.loads(ln))
    except Exception as e: errs.append(f"line {i+1} parse: {e}")

for r in recs:
    sid = r.get("source_id","?")
    for fld in FIELDS:
        if fld not in r: errs.append(f"{sid}: missing {fld}")
    if r.get("teacher_lane") != "teacher-B": errs.append(f"{sid}: bad teacher_lane")
    if r.get("teacher_model") != "claude-opus-5-current": errs.append(f"{sid}: bad teacher_model")
    if r.get("calibration_status") != "provisional": errs.append(f"{sid}: bad calibration_status")
    if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{sid}: bad decision")
    src = tr_map.get(sid) or va_map.get(sid)
    if not src: errs.append(f"{sid}: not in corpus")
    else:
        if r.get("source_user") != src[1]: errs.append(f"{sid}: source_user mismatch")
        if r.get("source_assistant") != src[2]: errs.append(f"{sid}: source_assistant mismatch")
    ca = r.get("corrected_answer")
    if not isinstance(ca,str) or not ca.strip(): errs.append(f"{sid}: empty corrected_answer")
    qd = r.get("quality_dimensions")
    if not isinstance(qd,dict): errs.append(f"{sid}: quality_dimensions not object")
    else:
        for k in ("technical_correctness","instruction_coverage","operational_safety"):
            v = qd.get(k)
            if not isinstance(v,int) or isinstance(v,bool) or not 1<=v<=5:
                errs.append(f"{sid}: bad {k}={v}")
    for k in ("risks","evidence_required"):
        v = r.get(k)
        if not isinstance(v,list) or not all(isinstance(x,str) for x in v):
            errs.append(f"{sid}: {k} not list[str]")
    c = r.get("confidence")
    if not isinstance(c,(int,float)) or isinstance(c,bool) or not 0.0<=c<=1.0:
        errs.append(f"{sid}: bad confidence {c}")

# --- global aggregate checks ---
def agg(prefix):
    ids = []
    for p in sorted(glob.glob(f"{RES}/{prefix}-batch-*.jsonl")):
        for ln in open(p):
            ln = ln.strip()
            if ln: ids.append(json.loads(ln)["source_id"])
    return ids

tr_ids, va_ids = agg("train"), agg("validation")
allids = tr_ids + va_ids
if len(allids) != len(set(allids)): errs.append("duplicate source_id globally")
exp_tr = [c[0] for c in TR][:len(tr_ids)]
if tr_ids != exp_tr: errs.append("train sequence is not a strict corpus prefix")
exp_va = [c[0] for c in VA][:len(va_ids)]
if va_ids != exp_va: errs.append("validation sequence is not a strict corpus prefix")

print("train_total", len(tr_ids), "validation_total", len(va_ids), "grand_total", len(allids))
if errs:
    print("VERIFY_FAIL", len(errs))
    for e in errs[:40]: print(" -", e)
    sys.exit(1)
print("VERIFY_PASS")
