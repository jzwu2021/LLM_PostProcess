import json, glob, os, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
BATCH = f"{EXP}/results/train-batch-0237.jsonl"
START, END = 2360, 2370

REQ = {"source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"}

fail = []
def chk(c, msg):
    if not c: fail.append(msg)

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]

raw = open(BATCH).read()
lines = raw.split("\n")
chk(lines[-1] == "", "batch file must end with newline")
lines = [l for l in lines if l.strip()]
rows = []
for i, l in enumerate(lines):
    try:
        rows.append(json.loads(l))
    except Exception as e:
        fail.append(f"line {i+1} not parseable: {e}")

chk(len(rows) == 10, f"batch count {len(rows)} != 10")

src = corpus[START:END]
for i, r in enumerate(rows):
    got = set(r.keys())
    chk(got == REQ, f"row {i} field set mismatch: extra={got-REQ} missing={REQ-got}")
    chk(r.get("teacher_lane") == "teacher-B", f"row {i} teacher_lane")
    chk(r.get("teacher_model") == "claude-opus-5-current", f"row {i} teacher_model")
    chk(r.get("calibration_status") == "provisional", f"row {i} calibration_status")
    chk(r.get("decision") in ("keep","rewrite","reject"), f"row {i} decision")
    s = src[i]
    m = {x["role"]: x["content"] for x in s["messages"]}
    chk(r.get("source_id") == s["id"], f"row {i} source_id positional mismatch {r.get('source_id')} vs {s['id']}")
    chk(r.get("source_user") == m["user"], f"row {i} source_user not byte-equal")
    chk(r.get("source_assistant") == m["assistant"], f"row {i} source_assistant not byte-equal")
    ca = r.get("corrected_answer")
    chk(isinstance(ca, str) and ca.strip() != "", f"row {i} corrected_answer empty")
    qd = r.get("quality_dimensions")
    chk(isinstance(qd, dict) and set(qd.keys()) == {"technical_correctness","instruction_coverage","operational_safety"},
        f"row {i} quality_dimensions keys")
    if isinstance(qd, dict):
        for k, v in qd.items():
            chk(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5, f"row {i} qd {k}={v!r}")
    for fld in ("risks","evidence_required"):
        v = r.get(fld)
        chk(isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) and x.strip() for x in v),
            f"row {i} {fld} bad")
    c = r.get("confidence")
    chk(isinstance(c, float) and not isinstance(c, bool) and 0.0 <= c <= 1.0, f"row {i} confidence {c!r}")

# stance uniqueness within this batch
heads = [r["corrected_answer"].split("\n",1)[0] for r in rows]
chk(len(set(heads)) == len(heads), "duplicate stance headers in this batch")

# global aggregate: prefix property + id uniqueness (historical: non-interference only)
files = sorted(glob.glob(f"{EXP}/results/train-batch-*.jsonl"))
agg = []
for f in files:
    for l in open(f):
        if l.strip(): agg.append(json.loads(l))
ids = [r["source_id"] for r in agg]
chk(len(set(ids)) == len(ids), "duplicate source_id in aggregate")
corpus_ids = [r["id"] for r in corpus]
chk(ids == corpus_ids[:len(ids)], "aggregate train sequence is not a strict prefix of train.jsonl")
chk(len(agg) == 2370, f"aggregate count {len(agg)} != 2370")

# no validation batches ever
chk(len(glob.glob(f"{EXP}/results/validation-batch-*.jsonl")) == 0, "validation batch files must not exist")

if fail:
    print("VERIFY_FAIL")
    for f in fail[:40]: print(" -", f)
    sys.exit(1)
print("VERIFY_PASS batch=10 aggregate=%d prefix=ok ids_unique=ok" % len(agg))
