#!/usr/bin/env python3
"""Independent ad-hoc verification for teacher-B train-batch-0167."""
import json, os, glob, re, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
RES = os.path.join(EXP, "results")
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
BATCH = os.path.join(RES, "train-batch-0167.jsonl")
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
fails = []

raw = open(BATCH, "rb").read().decode("utf-8")
lines = raw.split("\n")
assert lines[-1] == "", "file must end with newline"
lines = lines[:-1]
if len(lines) != 10:
    fails.append(f"batch line count {len(lines)} != 10")
recs = []
for i, l in enumerate(lines):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fails.append(f"line {i+1} not parseable: {e}")

corpus = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
cmap = {}
for r in corpus:
    u = next(m["content"] for m in r["messages"] if m["role"] == "user")
    a = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
    cmap[r["id"]] = (u, a)

for r in recs:
    sid = r.get("source_id")
    for k in REQ:
        if k not in r:
            fails.append(f"{sid}: missing field {k}")
    if r.get("teacher_lane") != "teacher-B": fails.append(f"{sid}: bad lane")
    if r.get("teacher_model") != "claude-opus-5-current": fails.append(f"{sid}: bad model")
    if r.get("calibration_status") != "provisional": fails.append(f"{sid}: bad status")
    if r.get("decision") not in ("keep","rewrite","reject"): fails.append(f"{sid}: bad decision")
    if sid not in cmap:
        fails.append(f"{sid}: not in corpus")
    else:
        u, a = cmap[sid]
        if r.get("source_user") != u: fails.append(f"{sid}: source_user mismatch")
        if r.get("source_assistant") != a: fails.append(f"{sid}: source_assistant mismatch")
    ca = r.get("corrected_answer")
    if not isinstance(ca, str) or not ca.strip(): fails.append(f"{sid}: empty corrected_answer")
    qd = r.get("quality_dimensions")
    if not isinstance(qd, dict): fails.append(f"{sid}: qd not object")
    else:
        for d in ("technical_correctness","instruction_coverage","operational_safety"):
            v = qd.get(d)
            if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= 5):
                fails.append(f"{sid}: bad qd {d}={v}")
    if not isinstance(r.get("risks"), list) or not all(isinstance(x,str) for x in r["risks"]):
        fails.append(f"{sid}: risks not str list")
    if not isinstance(r.get("evidence_required"), list) or not all(isinstance(x,str) for x in r["evidence_required"]):
        fails.append(f"{sid}: evidence_required not str list")
    c = r.get("confidence")
    if not isinstance(c, float) or not (0.0 <= c <= 1.0): fails.append(f"{sid}: bad confidence {c}")

# anti-template: corrected_answer must be unique within batch
if len({r["corrected_answer"] for r in recs}) != len(recs):
    fails.append("duplicate corrected_answer within batch")

# global aggregation: prefix of corpus + global unique ids
files = sorted(glob.glob(os.path.join(RES, "train-batch-*.jsonl")))
nums = [int(re.search(r"train-batch-(\d{4})\.jsonl$", f).group(1)) for f in files]
if nums != list(range(1, len(nums)+1)):
    fails.append(f"batch numbering not contiguous from 0001: {nums[:3]}..{nums[-3:]}")
allids = []
for f in files:
    for l in open(f, encoding="utf-8"):
        l = l.strip()
        if l:
            allids.append(json.loads(l)["source_id"])
if len(allids) != len(set(allids)):
    fails.append("duplicate source_id globally")
corpus_ids = [r["id"] for r in corpus]
if allids != corpus_ids[:len(allids)]:
    for i,(x,y) in enumerate(zip(allids, corpus_ids)):
        if x != y:
            fails.append(f"prefix mismatch at index {i}: {x} != {y}")
            break
    else:
        fails.append("prefix length mismatch")

print("total_train_processed =", len(allids))
print("batch_records =", len(recs))
print("FAILS:", len(fails))
for f in fails[:20]:
    print("  -", f)
print("VERIFY_RESULT =", "PASS" if not fails else "FAIL")
sys.exit(1 if fails else 0)
