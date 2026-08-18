import json, os, glob, sys

EXP = "experiments/2026-08-17-teacher-b-corpus-review"
BATCH = os.path.join(EXP, "results", "train-batch-0175.jsonl")
errs = []

corpus = [json.loads(l) for l in open("research/ai-infra-expert/corpus/train.jsonl")]
cmap = {}
for r in corpus:
    m = {x["role"]: x["content"] for x in r["messages"]}
    cmap[r["id"]] = (m["user"], m["assistant"])
order = [r["id"] for r in corpus]

REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]

recs = []
for i, line in enumerate(open(BATCH), 1):
    line = line.rstrip("\n")
    try:
        recs.append(json.loads(line))
    except Exception as e:
        errs.append(f"line {i} parse: {e}")
if len(recs) != 10:
    errs.append(f"batch count {len(recs)} != 10")

for r in recs:
    sid = r.get("source_id")
    for k in REQ:
        if k not in r: errs.append(f"{sid} missing {k}")
    if r.get("teacher_lane") != "teacher-B": errs.append(f"{sid} lane")
    if r.get("teacher_model") != "claude-opus-5-current": errs.append(f"{sid} model")
    if r.get("calibration_status") != "provisional": errs.append(f"{sid} status")
    if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{sid} decision")
    if sid not in cmap:
        errs.append(f"{sid} not in corpus"); continue
    u, a = cmap[sid]
    if r.get("source_user") != u: errs.append(f"{sid} source_user mismatch")
    if r.get("source_assistant") != a: errs.append(f"{sid} source_assistant mismatch")
    ca = r.get("corrected_answer")
    if not isinstance(ca, str) or not ca.strip(): errs.append(f"{sid} empty corrected_answer")
    qd = r.get("quality_dimensions", {})
    for d in ("technical_correctness","instruction_coverage","operational_safety"):
        v = qd.get(d)
        if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= 5):
            errs.append(f"{sid} qd {d}={v}")
    if not isinstance(r.get("risks"), list) or not all(isinstance(x,str) for x in r["risks"]):
        errs.append(f"{sid} risks")
    if not isinstance(r.get("evidence_required"), list) or not all(isinstance(x,str) for x in r["evidence_required"]):
        errs.append(f"{sid} evidence_required")
    c = r.get("confidence")
    if not isinstance(c,(int,float)) or isinstance(c,bool) or not (0.0 <= c <= 1.0):
        errs.append(f"{sid} confidence {c}")

# global aggregate: uniqueness + strict prefix
files = sorted(glob.glob(os.path.join(EXP,"results","train-batch-*.jsonl")))
allids = []
for f in files:
    for line in open(f):
        allids.append(json.loads(line)["source_id"])
if len(allids) != len(set(allids)):
    from collections import Counter
    dup = [k for k,v in Counter(allids).items() if v>1]
    errs.append(f"duplicate source_ids: {dup[:10]}")
if allids != order[:len(allids)]:
    bad = next(i for i,(x,y) in enumerate(zip(allids, order)) if x!=y)
    errs.append(f"prefix mismatch at index {bad}: {allids[bad]} vs {order[bad]}")

print("batch_files:", len(files))
print("train_total:", len(allids))
print("this_batch:", len(recs))
print("decisions:", {d: sum(1 for r in recs if r["decision"]==d) for d in ("keep","rewrite","reject")})
print("VERIFY:", "PASS" if not errs else "FAIL")
for e in errs[:30]: print("  -", e)
sys.exit(1 if errs else 0)
