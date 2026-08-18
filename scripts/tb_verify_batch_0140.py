#!/usr/bin/env python3
"""Ad-hoc verification for teacher-B review outputs (this run: train-batch-0140)."""
import json, glob, os, re, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
RES = os.path.join(EXP, "results")
BATCH = os.path.join(RES, "train-batch-0140.jsonl")
CORPUS = {
    "train": os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl"),
    "validation": os.path.join(ROOT, "research/ai-infra-expert/corpus/validation.jsonl"),
}
REQ = ["source_id", "teacher_lane", "teacher_model", "calibration_status", "decision",
       "source_user", "source_assistant", "corrected_answer", "quality_dimensions",
       "risks", "evidence_required", "confidence"]
errs = []


def load_corpus(p):
    seq = []
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        ms = d["messages"]
        seq.append((d["id"],
                    [m["content"] for m in ms if m["role"] == "user"][0],
                    [m["content"] for m in ms if m["role"] == "assistant"][0]))
    return seq


corp = {k: load_corpus(v) for k, v in CORPUS.items()}

# 1) this batch parses line-by-line, correct count
recs = []
with open(BATCH, encoding="utf-8") as f:
    raw = f.read()
if not raw.endswith("\n"):
    errs.append("batch file does not end with newline")
for i, line in enumerate(raw.splitlines(), 1):
    try:
        recs.append(json.loads(line))
    except Exception as e:
        errs.append(f"line {i} not valid JSON: {e}")
if len(recs) != 10:
    errs.append(f"batch record count {len(recs)} != 10")

# 2) per-record schema
for r in recs:
    sid = r.get("source_id", "?")
    for k in REQ:
        if k not in r:
            errs.append(f"{sid}: missing field {k}")
    if r.get("teacher_lane") != "teacher-B":
        errs.append(f"{sid}: bad teacher_lane")
    if r.get("teacher_model") != "claude-opus-5-current":
        errs.append(f"{sid}: bad teacher_model")
    if r.get("calibration_status") != "provisional":
        errs.append(f"{sid}: bad calibration_status")
    if r.get("decision") not in ("keep", "rewrite", "reject"):
        errs.append(f"{sid}: bad decision {r.get('decision')}")
    ca = r.get("corrected_answer")
    if not isinstance(ca, str) or not ca.strip():
        errs.append(f"{sid}: empty corrected_answer")
    c = r.get("confidence")
    if not isinstance(c, (int, float)) or isinstance(c, bool) or not (0.0 <= c <= 1.0):
        errs.append(f"{sid}: confidence out of range: {c}")
    qd = r.get("quality_dimensions")
    if not isinstance(qd, dict):
        errs.append(f"{sid}: quality_dimensions not object")
    else:
        for k in ("technical_correctness", "instruction_coverage", "operational_safety"):
            v = qd.get(k)
            if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= 5):
                errs.append(f"{sid}: quality_dimensions.{k} invalid: {v}")
    for k in ("risks", "evidence_required"):
        v = r.get(k)
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v) or not v:
            errs.append(f"{sid}: {k} must be non-empty list of strings")

# 3) aggregate all batches: prefix check + global uniqueness
seen = {}
for split in ("train", "validation"):
    files = sorted(glob.glob(os.path.join(RES, f"{split}-batch-*.jsonl")))
    nums = [int(re.search(r"-(\d{4})\.jsonl$", f).group(1)) for f in files]
    if nums and nums != list(range(1, len(nums) + 1)):
        errs.append(f"{split}: batch numbering not contiguous from 0001: {nums[:3]}...{nums[-3:]}")
    agg = []
    for fp in files:
        for i, line in enumerate(open(fp, encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception as e:
                errs.append(f"{os.path.basename(fp)}:{i} bad JSON: {e}")
                continue
            agg.append(d)
    for idx, d in enumerate(agg):
        sid = d.get("source_id")
        if sid in seen:
            errs.append(f"duplicate source_id {sid} (also in {seen[sid]})")
        else:
            seen[sid] = split
        if idx >= len(corp[split]):
            errs.append(f"{split}: index {idx} beyond corpus length")
            continue
        cid, cu, ca_ = corp[split][idx]
        if sid != cid:
            errs.append(f"{split}[{idx}]: source_id {sid} != corpus {cid} (prefix order violated)")
        if d.get("source_user") != cu:
            errs.append(f"{sid}: source_user mismatch vs corpus")
        if d.get("source_assistant") != ca_:
            errs.append(f"{sid}: source_assistant mismatch vs corpus")
    print(f"{split}: {len(agg)} records aggregated (corpus {len(corp[split])})")

print("total processed:", sum(1 for _ in seen))
if errs:
    print("VERIFY_FAIL")
    for e in errs[:60]:
        print(" -", e)
    sys.exit(1)
print("VERIFY_PASS")
