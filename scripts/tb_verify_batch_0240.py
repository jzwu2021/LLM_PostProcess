import json, os, glob, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
RES = os.path.join(EXP, "results")
fail = []

def corpus(path):
    out = []
    for l in open(path):
        d = json.loads(l)
        u = [m for m in d["messages"] if m["role"] == "user"][0]["content"]
        a = [m for m in d["messages"] if m["role"] == "assistant"][0]["content"]
        out.append((d["id"], u, a))
    return out

train = corpus(os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl"))
val = corpus(os.path.join(ROOT, "research/ai-infra-expert/corpus/validation.jsonl"))

REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]

def load(prefix):
    recs = []
    for f in sorted(glob.glob(os.path.join(RES, prefix + "-batch-*.jsonl"))):
        n = 0
        for ln, line in enumerate(open(f), 1):
            if not line.strip():
                fail.append(f"{f}:{ln} blank line"); continue
            try:
                r = json.loads(line)
            except Exception as e:
                fail.append(f"{f}:{ln} json parse: {e}"); continue
            recs.append((f, ln, r)); n += 1
        if n != 10:
            fail.append(f"{f} has {n} records, expected 10")
    return recs

def check(recs, ref, label):
    if len(recs) > len(ref):
        fail.append(f"{label}: {len(recs)} records exceed corpus {len(ref)}")
    for i, (f, ln, r) in enumerate(recs):
        for k in REQ:
            if k not in r:
                fail.append(f"{f}:{ln} missing field {k}")
        if [k for k in REQ if k not in r]:
            continue
        if r["teacher_lane"] != "teacher-B": fail.append(f"{f}:{ln} bad teacher_lane")
        if r["teacher_model"] != "claude-opus-5-current": fail.append(f"{f}:{ln} bad teacher_model")
        if r["calibration_status"] != "provisional": fail.append(f"{f}:{ln} bad calibration_status")
        if r["decision"] not in ("keep","rewrite","reject"): fail.append(f"{f}:{ln} bad decision")
        if not isinstance(r["corrected_answer"], str) or not r["corrected_answer"].strip():
            fail.append(f"{f}:{ln} empty corrected_answer")
        qd = r["quality_dimensions"]
        if not isinstance(qd, dict): fail.append(f"{f}:{ln} quality_dimensions not object")
        else:
            for d in ("technical_correctness","instruction_coverage","operational_safety"):
                v = qd.get(d)
                if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= 5):
                    fail.append(f"{f}:{ln} bad quality_dimensions.{d}={v!r}")
        for k in ("risks","evidence_required"):
            v = r[k]
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                fail.append(f"{f}:{ln} {k} not string array")
        c = r["confidence"]
        if not isinstance(c, (int, float)) or isinstance(c, bool) or not (0.0 <= c <= 1.0):
            fail.append(f"{f}:{ln} bad confidence {c!r}")
        if i < len(ref):
            cid, cu, ca = ref[i]
            if r["source_id"] != cid: fail.append(f"{f}:{ln} prefix order broken: {r['source_id']} != {cid} at idx {i}")
            if r["source_user"] != cu: fail.append(f"{f}:{ln} source_user mismatch for {cid}")
            if r["source_assistant"] != ca: fail.append(f"{f}:{ln} source_assistant mismatch for {cid}")

tr = load("train"); va = load("validation")
check(tr, train, "train"); check(va, val, "validation")

ids = [r["source_id"] for _, _, r in tr + va if "source_id" in r]
if len(ids) != len(set(ids)):
    seen = set(); dup = sorted({i for i in ids if i in seen or seen.add(i)})
    fail.append(f"duplicate source_id: {dup[:20]}")

print("train_records", len(tr))
print("validation_records", len(va))
print("total", len(tr) + len(va))
if fail:
    print("VERIFY_FAIL", len(fail))
    for f in fail[:40]: print("  ", f)
    sys.exit(1)
print("VERIFY_PASS")
