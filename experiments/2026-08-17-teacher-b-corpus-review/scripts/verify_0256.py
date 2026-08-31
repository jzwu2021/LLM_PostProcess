import json, glob, os, re, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
BATCH = f"{EXP}/results/train-batch-0256.jsonl"
START, END = 2550, 2560

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
    chk(isinstance(ca, str) and ca != r.get("source_assistant"), f"row {i} corrected_answer equals source_assistant")
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

# content contract: each rewrite carries the required review paragraphs and an explicit evidence label
for i, r in enumerate(rows):
    ca = r.get("corrected_answer") or ""
    for para in ("Mechanism.", "Falsifiable hypothesis.", "Metrics.", "Controlled experiment.",
                 "Confounders.", "Rollback criteria."):
        chk(para in ca, f"row {i} missing paragraph {para}")
    chk("H1:" in ca, f"row {i} missing numbered hypothesis H1")
    chk("Falsified if" in ca, f"row {i} missing explicit falsification condition")
    chk("ESTIMATE" in ca or "MEASURED" in ca, f"row {i} missing ESTIMATE/MEASURED label")
    chk("teacher-A" not in ca and "teacher_a" not in ca, f"row {i} teacher-A leak")

# stance uniqueness within this batch
heads = [r["corrected_answer"].split("\n",1)[0] for r in rows]
chk(len(set(heads)) == len(heads), "duplicate stance headers in this batch")

# stance uniqueness against every previously committed batch
prior_heads = set()
for f in sorted(glob.glob(f"{EXP}/results/train-batch-*.jsonl")):
    if os.path.basename(f) == os.path.basename(BATCH): continue
    for l in open(f):
        if l.strip(): prior_heads.add(json.loads(l)["corrected_answer"].split("\n",1)[0])
overlap = prior_heads & set(heads)
chk(not overlap, f"stance headers reused from prior batches: {sorted(overlap)[:3]}")

prior_nums = set()
for h in prior_heads:
    m = re.match(r"STANCE (\d+)", h)
    if m: prior_nums.add(int(m.group(1)))
new_nums = set()
for h in heads:
    m = re.match(r"STANCE (\d+)", h)
    if m: new_nums.add(int(m.group(1)))
chk(len(new_nums) == 10, f"expected 10 numbered stances, got {len(new_nums)}")
chk(not (prior_nums & new_nums), f"stance numbers reused: {sorted(prior_nums & new_nums)}")

# global aggregate: prefix property + id uniqueness
files = sorted(glob.glob(f"{EXP}/results/train-batch-*.jsonl"))
agg = []
for f in files:
    for l in open(f):
        if l.strip(): agg.append(json.loads(l))
ids = [r["source_id"] for r in agg]
chk(len(set(ids)) == len(ids), "duplicate source_id in aggregate")
corpus_ids = [r["id"] for r in corpus]
chk(ids == corpus_ids[:len(ids)], "aggregate train sequence is not a strict prefix of train.jsonl")
chk(len(agg) == 2560, f"aggregate count {len(agg)} != 2560")

# no validation batches ever
chk(len(glob.glob(f"{EXP}/results/validation-batch-*.jsonl")) == 0, "validation batch files must not exist")

if fail:
    print("VERIFY_FAIL")
    for f in fail[:40]: print(" -", f)
    sys.exit(1)
print("VERIFY_PASS batch=10 aggregate=%d prefix=ok ids_unique=ok stances=ok" % len(agg))
