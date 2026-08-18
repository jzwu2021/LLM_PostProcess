#!/usr/bin/env python3
"""Ad-hoc verification for teacher-B train batches through 0170."""
import json, os, glob, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
BATCH = os.path.join(EXP, "results/train-batch-0170.jsonl")
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
fail = []
def chk(c, m):
    if not c: fail.append(m)

corpus = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
cmap = {r["id"]: r for r in corpus}

lines = open(BATCH, encoding="utf-8").read().split("\n")
lines = [l for l in lines if l.strip()]
chk(len(lines) == 10, f"batch line count {len(lines)} != 10")
recs = []
for i, l in enumerate(lines):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fail.append(f"line {i+1} not JSON: {e}")

for r in recs:
    sid = r.get("source_id")
    chk(set(REQ) <= set(r.keys()), f"{sid} missing fields {set(REQ)-set(r.keys())}")
    chk(r.get("teacher_lane") == "teacher-B", f"{sid} bad lane")
    chk(r.get("teacher_model") == "claude-opus-5-current", f"{sid} bad model")
    chk(r.get("calibration_status") == "provisional", f"{sid} bad status")
    chk(r.get("decision") in ("keep","rewrite","reject"), f"{sid} bad decision")
    src = cmap.get(sid)
    chk(src is not None, f"{sid} not in corpus")
    if src:
        u = next(m["content"] for m in src["messages"] if m["role"]=="user")
        a = next(m["content"] for m in src["messages"] if m["role"]=="assistant")
        chk(r["source_user"] == u, f"{sid} source_user mismatch")
        chk(r["source_assistant"] == a, f"{sid} source_assistant mismatch")
    chk(isinstance(r.get("corrected_answer"), str) and r["corrected_answer"].strip(), f"{sid} empty corrected_answer")
    qd = r.get("quality_dimensions", {})
    for k in ("technical_correctness","instruction_coverage","operational_safety"):
        chk(isinstance(qd.get(k), int) and 1 <= qd[k] <= 5, f"{sid} bad {k}")
    chk(isinstance(r.get("risks"), list) and all(isinstance(x,str) for x in r["risks"]), f"{sid} bad risks")
    chk(isinstance(r.get("evidence_required"), list) and all(isinstance(x,str) for x in r["evidence_required"]), f"{sid} bad evidence")
    c = r.get("confidence")
    chk(isinstance(c,(int,float)) and 0.0 <= c <= 1.0, f"{sid} bad confidence")

# global aggregation
allb = sorted(glob.glob(os.path.join(EXP, "results/train-batch-*.jsonl")))
ids = []
for b in allb:
    for l in open(b, encoding="utf-8"):
        if l.strip():
            ids.append(json.loads(l)["source_id"])
chk(len(ids) == len(set(ids)), "duplicate source_id across batches")
cids = [r["id"] for r in corpus]
chk(ids == cids[:len(ids)], "aggregated train sequence is not a strict prefix of train.jsonl")

vb = sorted(glob.glob(os.path.join(EXP, "results/validation-batch-*.jsonl")))
chk(not vb, f"unexpected validation batches: {vb}")

print("total_train_processed =", len(ids))
if fail:
    print("VERIFY_FAIL")
    for f_ in fail: print("  -", f_)
    sys.exit(1)
print("VERIFY_PASS")
