#!/usr/bin/env python3
"""Independent ad-hoc verification for teacher-B review batch 0144 (written fresh this run)."""
import json, os, sys, glob, re

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
RES = os.path.join(EXP, "results")
BATCH = os.path.join(RES, "train-batch-0144.jsonl")
REQ = ["source_id", "teacher_lane", "teacher_model", "calibration_status", "decision",
       "source_user", "source_assistant", "corrected_answer", "quality_dimensions",
       "risks", "evidence_required", "confidence"]
fails = []


def corpus(split):
    p = os.path.join(ROOT, "research/ai-infra-expert/corpus/%s.jsonl" % split)
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            m = {x["role"]: x["content"] for x in d["messages"]}
            out.append((d["id"], m["user"], m["assistant"]))
    return out


tr, va = corpus("train"), corpus("validation")

raw = open(BATCH, "rb").read().decode("utf-8")
lines = raw.split("\n")
if lines[-1] != "":
    fails.append("batch file does not end with newline")
lines = [l for l in lines if l != ""]
if len(lines) != 10:
    fails.append("batch line count %d != 10" % len(lines))
recs = []
for i, l in enumerate(lines, 1):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fails.append("line %d unparseable: %s" % (i, e))

idx = {cid: (u, a) for cid, u, a in tr}
for r in recs:
    sid = r.get("source_id")
    for k in REQ:
        if k not in r:
            fails.append("%s missing field %s" % (sid, k))
    if r.get("teacher_lane") != "teacher-B":
        fails.append("%s bad teacher_lane" % sid)
    if r.get("teacher_model") != "claude-opus-5-current":
        fails.append("%s bad teacher_model" % sid)
    if r.get("calibration_status") != "provisional":
        fails.append("%s bad calibration_status" % sid)
    if r.get("decision") not in ("keep", "rewrite", "reject"):
        fails.append("%s bad decision" % sid)
    if sid not in idx:
        fails.append("%s not in train corpus" % sid)
    else:
        u, a = idx[sid]
        if r.get("source_user") != u:
            fails.append("%s source_user mismatch" % sid)
        if r.get("source_assistant") != a:
            fails.append("%s source_assistant mismatch" % sid)
    ca = r.get("corrected_answer")
    if not isinstance(ca, str) or not ca.strip():
        fails.append("%s empty corrected_answer" % sid)
    qd = r.get("quality_dimensions")
    if not isinstance(qd, dict):
        fails.append("%s quality_dimensions not object" % sid)
    else:
        for d in ("technical_correctness", "instruction_coverage", "operational_safety"):
            v = qd.get(d)
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 5:
                fails.append("%s bad %s=%r" % (sid, d, v))
    for lk in ("risks", "evidence_required"):
        v = r.get(lk)
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x.strip() for x in v):
            fails.append("%s bad %s" % (sid, lk))
    c = r.get("confidence")
    if not isinstance(c, (int, float)) or isinstance(c, bool) or not 0.0 <= c <= 1.0:
        fails.append("%s bad confidence=%r" % (sid, c))

# global aggregate: uniqueness + prefix property
def agg(prefix):
    seq = []
    for p in sorted(glob.glob(os.path.join(RES, prefix + "-batch-*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seq.append(json.loads(line)["source_id"])
    return seq

tseq, vseq = agg("train"), agg("validation")
allids = tseq + vseq
if len(allids) != len(set(allids)):
    fails.append("duplicate source_id across all batches")
if tseq != [c[0] for c in tr[:len(tseq)]]:
    fails.append("train sequence is not a strict prefix of train corpus")
if vseq != [c[0] for c in va[:len(vseq)]]:
    fails.append("validation sequence is not a strict prefix of validation corpus")

print("batch=train-batch-0144 records=%d train_total=%d val_total=%d grand_total=%d"
      % (len(recs), len(tseq), len(vseq), len(allids)))
print("FAILURES=%d" % len(fails))
for f in fails[:40]:
    print("  FAIL:", f)
sys.exit(1 if fails else 0)
