#!/usr/bin/env python3
"""Ad-hoc verification for teacher-B batch 0169 + full aggregate prefix check."""
import json, glob, os, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
BATCH = f"{EXP}/results/train-batch-0169.jsonl"
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
errs = []
def chk(c, m):
    if not c: errs.append(m)

# corpus index
corpus = []
with open(CORPUS) as f:
    for l in f:
        d = json.loads(l)
        m = d["messages"]
        corpus.append((d["id"],
                       [x for x in m if x["role"]=="user"][0]["content"],
                       [x for x in m if x["role"]=="assistant"][0]["content"]))
cmap = {c[0]: c for c in corpus}

raw = open(BATCH, "rb").read().decode()
lines = raw.split("\n")
chk(lines[-1] == "", "batch must end with newline")
lines = [l for l in lines if l]
chk(len(lines) == 10, f"batch line count {len(lines)} != 10")
batch = []
for i, l in enumerate(lines):
    try:
        batch.append(json.loads(l))
    except Exception as e:
        chk(False, f"line {i+1} not valid JSON: {e}")

for r in batch:
    sid = r.get("source_id")
    chk(sorted(r.keys()) == sorted(REQ), f"{sid}: field set mismatch {sorted(r.keys())}")
    chk(r.get("teacher_lane") == "teacher-B", f"{sid}: bad lane")
    chk(r.get("teacher_model") == "claude-opus-5-current", f"{sid}: bad model")
    chk(r.get("calibration_status") == "provisional", f"{sid}: bad status")
    chk(r.get("decision") in ("keep","rewrite","reject"), f"{sid}: bad decision")
    chk(sid in cmap, f"{sid}: not in corpus")
    if sid in cmap:
        chk(r["source_user"] == cmap[sid][1], f"{sid}: source_user mismatch")
        chk(r["source_assistant"] == cmap[sid][2], f"{sid}: source_assistant mismatch")
    ca = r.get("corrected_answer")
    chk(isinstance(ca, str) and ca.strip() != "", f"{sid}: empty corrected_answer")
    qd = r.get("quality_dimensions")
    chk(isinstance(qd, dict) and sorted(qd.keys()) == ["instruction_coverage","operational_safety","technical_correctness"],
        f"{sid}: bad quality_dimensions keys")
    if isinstance(qd, dict):
        for k, v in qd.items():
            chk(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5, f"{sid}: qd {k}={v}")
    chk(isinstance(r.get("risks"), list) and all(isinstance(x,str) for x in r["risks"]) and r["risks"], f"{sid}: risks")
    chk(isinstance(r.get("evidence_required"), list) and all(isinstance(x,str) for x in r["evidence_required"]) and r["evidence_required"], f"{sid}: evidence_required")
    cf = r.get("confidence")
    chk(isinstance(cf, float) and 0.0 <= cf <= 1.0, f"{sid}: confidence {cf}")

# aggregate
files = sorted(glob.glob(f"{EXP}/results/train-batch-*.jsonl"))
agg = []
for fp in files:
    for l in open(fp):
        if l.strip(): agg.append(json.loads(l)["source_id"])
chk(len(agg) == len(set(agg)), "duplicate source_id in aggregate")
pref = [c[0] for c in corpus[:len(agg)]]
chk(agg == pref, "aggregate is NOT a strict prefix of train.jsonl")
# no validation files
vf = glob.glob(f"{EXP}/results/validation-batch-*.jsonl")
chk(not vf, f"validation batch files present: {vf}")

print(f"files={len(files)} aggregate={len(agg)} batch_records={len(batch)} last={agg[-1]}")
if errs:
    print("VERIFY_FAIL")
    for e in errs[:30]: print(" -", e)
    sys.exit(1)
print("VERIFY_PASS")
