import json, os, glob, sys

EXP = "experiments/2026-08-17-teacher-b-corpus-review"
RES = os.path.join(EXP, "results")
REQ = ["source_id", "teacher_lane", "teacher_model", "calibration_status", "decision",
       "source_user", "source_assistant", "corrected_answer", "quality_dimensions",
       "risks", "evidence_required", "confidence"]
QD = ["technical_correctness", "instruction_coverage", "operational_safety"]
fails = []


def load_corpus(p):
    out = []
    for line in open(p):
        d = json.loads(line)
        m = d["messages"]
        out.append((d["id"],
                    [x["content"] for x in m if x["role"] == "user"][0],
                    [x["content"] for x in m if x["role"] == "assistant"][0]))
    return out


corp = {"train": load_corpus("research/ai-infra-expert/corpus/train.jsonl"),
        "validation": load_corpus("research/ai-infra-expert/corpus/validation.jsonl")}

seq = {"train": [], "validation": []}
allids = []
for split in ("train", "validation"):
    for fp in sorted(glob.glob(os.path.join(RES, f"{split}-batch-*.jsonl"))):
        n = 0
        for i, line in enumerate(open(fp), 1):
            line = line.rstrip("\n")
            if not line:
                fails.append(f"{fp}:{i} empty line")
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                fails.append(f"{fp}:{i} JSON parse: {e}")
                continue
            n += 1
            for k in REQ:
                if k not in r:
                    fails.append(f"{fp}:{i} missing field {k}")
            if r.get("teacher_lane") != "teacher-B":
                fails.append(f"{fp}:{i} bad teacher_lane")
            if r.get("teacher_model") != "claude-opus-5-current":
                fails.append(f"{fp}:{i} bad teacher_model")
            if r.get("calibration_status") != "provisional":
                fails.append(f"{fp}:{i} bad calibration_status")
            if r.get("decision") not in ("keep", "rewrite", "reject"):
                fails.append(f"{fp}:{i} bad decision {r.get('decision')}")
            ca = r.get("corrected_answer")
            if not isinstance(ca, str) or not ca.strip():
                fails.append(f"{fp}:{i} empty corrected_answer")
            c = r.get("confidence")
            if not isinstance(c, (int, float)) or not (0.0 <= c <= 1.0):
                fails.append(f"{fp}:{i} confidence out of range")
            qd = r.get("quality_dimensions")
            if not isinstance(qd, dict):
                fails.append(f"{fp}:{i} quality_dimensions not object")
            else:
                for k in QD:
                    v = qd.get(k)
                    if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= 5):
                        fails.append(f"{fp}:{i} bad quality_dimensions.{k}")
            for k in ("risks", "evidence_required"):
                v = r.get(k)
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    fails.append(f"{fp}:{i} {k} not string array")
            allids.append(r.get("source_id"))
            seq[split].append((r.get("source_id"), r.get("source_user"), r.get("source_assistant")))
        if n != 10:
            fails.append(f"{fp} has {n} records, expected 10")

if len(allids) != len(set(allids)):
    dup = [x for x in set(allids) if allids.count(x) > 1]
    fails.append(f"duplicate source_id: {dup[:10]}")

for split in ("train", "validation"):
    got, exp = seq[split], corp[split]
    if len(got) > len(exp):
        fails.append(f"{split}: more records ({len(got)}) than corpus ({len(exp)})")
    for i, (g, e) in enumerate(zip(got, exp)):
        if g[0] != e[0]:
            fails.append(f"{split}[{i}] id mismatch {g[0]} != {e[0]}")
            break
        if g[1] != e[1]:
            fails.append(f"{split}[{i}] {g[0]} source_user mismatch")
        if g[2] != e[2]:
            fails.append(f"{split}[{i}] {g[0]} source_assistant mismatch")

print(f"train={len(seq['train'])}/{len(corp['train'])} validation={len(seq['validation'])}/{len(corp['validation'])} total={len(allids)}")
if fails:
    print("VERIFY_FAIL")
    for f in fails[:40]:
        print("  -", f)
    sys.exit(1)
print("VERIFY_PASS")
