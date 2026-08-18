#!/usr/bin/env python3
"""Independent ad-hoc verification for teacher-B train batches through 0165."""
import json, glob, os, re, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
BATCH = os.path.join(RES, "train-batch-0165.jsonl")
REQ = ["source_id", "teacher_lane", "teacher_model", "calibration_status", "decision",
       "source_user", "source_assistant", "corrected_answer", "quality_dimensions",
       "risks", "evidence_required", "confidence"]
fails = []


def chk(c, m):
    if not c:
        fails.append(m)


raw = open(BATCH, "rb").read().decode("utf-8")
lines = raw.split("\n")
chk(lines[-1] == "", "batch file must end with newline")
lines = [l for l in lines if l != ""]
chk(len(lines) == 10, f"expected 10 lines, got {len(lines)}")
recs = []
for i, l in enumerate(lines):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fails.append(f"line {i+1} not JSON: {e}")

corpus = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
cmap = {r["id"]: r for r in corpus}

for r in recs:
    sid = r.get("source_id")
    chk(sorted(r.keys()) == sorted(REQ), f"{sid}: field set mismatch {sorted(set(r)^set(REQ))}")
    chk(r.get("teacher_lane") == "teacher-B", f"{sid}: lane")
    chk(r.get("teacher_model") == "claude-opus-5-current", f"{sid}: model")
    chk(r.get("calibration_status") == "provisional", f"{sid}: status")
    chk(r.get("decision") in ("keep", "rewrite", "reject"), f"{sid}: decision")
    src = cmap.get(sid)
    chk(src is not None, f"{sid}: not in corpus")
    if src:
        su = next(m["content"] for m in src["messages"] if m["role"] == "user")
        sa = next(m["content"] for m in src["messages"] if m["role"] == "assistant")
        chk(r["source_user"] == su, f"{sid}: source_user mismatch")
        chk(r["source_assistant"] == sa, f"{sid}: source_assistant mismatch")
    ca = r.get("corrected_answer")
    chk(isinstance(ca, str) and len(ca.strip()) > 0, f"{sid}: empty corrected_answer")
    chk(ca != r.get("source_assistant"), f"{sid}: corrected_answer == source_assistant")
    qd = r.get("quality_dimensions")
    chk(isinstance(qd, dict) and sorted(qd) == ["instruction_coverage", "operational_safety", "technical_correctness"],
        f"{sid}: qd keys")
    if isinstance(qd, dict):
        for k, v in qd.items():
            chk(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5, f"{sid}: qd {k}={v}")
    chk(isinstance(r.get("risks"), list) and all(isinstance(x, str) for x in r["risks"]) and r["risks"], f"{sid}: risks")
    chk(isinstance(r.get("evidence_required"), list) and all(isinstance(x, str) for x in r["evidence_required"]) and r["evidence_required"], f"{sid}: evidence_required")
    c = r.get("confidence")
    chk(isinstance(c, float) and 0.0 <= c <= 1.0, f"{sid}: confidence {c}")
    # anti-template: each answer must name a distinct primary hypothesis
    chk("Primary hypothesis under test:" in ca, f"{sid}: missing primary hypothesis marker")
    chk("ESTIMATE" in ca or "MEASURED" in ca, f"{sid}: missing ESTIMATE/MEASURED tagging")

# uniqueness of corrected_answer within batch
mechs = [re.search(r"Primary hypothesis under test: (.*?)\.\n", r["corrected_answer"]).group(1) for r in recs if r.get("corrected_answer")]
chk(len(set(mechs)) == len(mechs), "duplicate mechanisms within batch")
chk(len({r["corrected_answer"] for r in recs}) == len(recs), "duplicate corrected_answer within batch")

# global: aggregate order must be strict prefix of corpus, ids unique
files = sorted(glob.glob(os.path.join(RES, "train-batch-*.jsonl")))
agg = []
for fp in files:
    for l in open(fp, encoding="utf-8"):
        l = l.strip()
        if l:
            agg.append(json.loads(l)["source_id"])
chk(len(agg) == len(set(agg)), "duplicate source_id globally")
expected = [r["id"] for r in corpus[:len(agg)]]
chk(agg == expected, "aggregate train sequence is NOT a strict prefix of train.jsonl")

vfiles = glob.glob(os.path.join(RES, "validation-batch-*"))
chk(not vfiles, f"validation batch files must not exist: {vfiles}")

print(f"batches={len(files)} aggregate={len(agg)} this_batch={len(recs)} ids={recs[0]['source_id']}..{recs[-1]['source_id']}")
if fails:
    print("VERIFY_FAIL")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("VERIFY_PASS")
