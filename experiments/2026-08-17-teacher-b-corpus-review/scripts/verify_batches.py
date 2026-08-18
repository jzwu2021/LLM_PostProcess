#!/usr/bin/env python3
"""Ad-hoc verifier for teacher-B blind review batches. BLIND: never reads teacher-A."""
import json, os, sys, glob, re

ROOT = "/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
CORPUS = {
    "train": os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl"),
    "validation": os.path.join(ROOT, "research/ai-infra-expert/corpus/validation.jsonl"),
}
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
errs = []

def load_corpus(p):
    out = []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            msgs = d["messages"]
            out.append((d["id"],
                        next(m["content"] for m in msgs if m["role"]=="user"),
                        next(m["content"] for m in msgs if m["role"]=="assistant")))
    return out

corp = {k: load_corpus(v) for k, v in CORPUS.items()}
seen = {}
seq = {"train": [], "validation": []}

for split in ("train","validation"):
    files = sorted(glob.glob(os.path.join(RES, f"{split}-batch-*.jsonl")))
    for fp in files:
        raw = open(fp, "rb").read().decode("utf-8")
        lines = raw.split("\n")
        if lines and lines[-1] == "": lines.pop()
        if len(lines) != 10:
            errs.append(f"{os.path.basename(fp)}: expected 10 lines, got {len(lines)}")
        for ln, line in enumerate(lines, 1):
            try:
                r = json.loads(line)
            except Exception as e:
                errs.append(f"{os.path.basename(fp)}:{ln} JSON parse error: {e}")
                continue
            tag = f"{os.path.basename(fp)}:{ln}"
            for k in REQ:
                if k not in r: errs.append(f"{tag} missing field {k}")
            if r.get("teacher_lane") != "teacher-B": errs.append(f"{tag} bad teacher_lane")
            if r.get("teacher_model") != "claude-opus-5-current": errs.append(f"{tag} bad teacher_model")
            if r.get("calibration_status") != "provisional": errs.append(f"{tag} bad calibration_status")
            if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{tag} bad decision")
            ca = r.get("corrected_answer")
            if not isinstance(ca,str) or not ca.strip(): errs.append(f"{tag} empty corrected_answer")
            c = r.get("confidence")
            if not isinstance(c,(int,float)) or not (0.0 <= c <= 1.0): errs.append(f"{tag} confidence out of range")
            qd = r.get("quality_dimensions")
            if not isinstance(qd, dict):
                errs.append(f"{tag} quality_dimensions not object")
            else:
                for d3 in ("technical_correctness","instruction_coverage","operational_safety"):
                    v = qd.get(d3)
                    if not isinstance(v,int) or isinstance(v,bool) or not (1 <= v <= 5):
                        errs.append(f"{tag} bad quality_dimensions.{d3}")
            for lk in ("risks","evidence_required"):
                v = r.get(lk)
                if not isinstance(v,list) or not all(isinstance(x,str) for x in v):
                    errs.append(f"{tag} {lk} not list[str]")
            sid = r.get("source_id")
            if sid in seen: errs.append(f"{tag} duplicate source_id {sid} (also {seen[sid]})")
            else: seen[sid] = tag
            seq[split].append((sid, r.get("source_user"), r.get("source_assistant"), tag))

for split in ("train","validation"):
    got, ref = seq[split], corp[split]
    if len(got) > len(ref):
        errs.append(f"{split}: {len(got)} records exceeds corpus {len(ref)}")
    for i,(sid,su,sa,tag) in enumerate(got):
        if i >= len(ref): break
        rid,rsu,rsa = ref[i]
        if sid != rid: errs.append(f"{tag} prefix order mismatch at {split}[{i}]: {sid} != {rid}")
        if su != rsu: errs.append(f"{tag} source_user mismatch for {sid}")
        if sa != rsa: errs.append(f"{tag} source_assistant mismatch for {sid}")

print(f"train={len(seq['train'])}/{len(corp['train'])} validation={len(seq['validation'])}/{len(corp['validation'])} total={len(seq['train'])+len(seq['validation'])}")
if errs:
    print("VERIFY_FAIL", len(errs))
    for e in errs[:50]: print(" -", e)
    sys.exit(1)
print("VERIFY_PASS")
