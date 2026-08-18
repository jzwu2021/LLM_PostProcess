import json, os, re, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
RES = os.path.join(EXP, "results")
CORP = os.path.join(ROOT, "research/ai-infra-expert/corpus")
BATCH = os.path.join(RES, "train-batch-0152.jsonl")

fails = []
def chk(cond, msg):
    if not cond:
        fails.append(msg)

REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]

def load_corpus(name):
    seq = []
    with open(os.path.join(CORP, name), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            u = next(m["content"] for m in d["messages"] if m["role"] == "user")
            a = next(m["content"] for m in d["messages"] if m["role"] == "assistant")
            seq.append((d["id"], u, a))
    return seq

train = load_corpus("train.jsonl")
val = load_corpus("validation.jsonl")
cmap = {t[0]: t for t in train + val}

# --- this batch: physical-line JSONL parse ---
raw = open(BATCH, "rb").read().decode("utf-8")
lines = raw.split("\n")
chk(lines[-1] == "", "batch must end with trailing newline")
lines = [l for l in lines if l != ""]
chk(len(lines) == 10, f"batch line count = {len(lines)}, expected 10")
recs = []
for i, l in enumerate(lines, 1):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fails.append(f"line {i} not valid JSON: {e}")

for r in recs:
    sid = r.get("source_id", "?")
    for k in REQ:
        chk(k in r, f"{sid}: missing field {k}")
    chk(r.get("teacher_lane") == "teacher-B", f"{sid}: bad teacher_lane")
    chk(r.get("teacher_model") == "claude-opus-5-current", f"{sid}: bad teacher_model")
    chk(r.get("calibration_status") == "provisional", f"{sid}: bad calibration_status")
    chk(r.get("decision") in ("keep","rewrite","reject"), f"{sid}: bad decision")
    ca = r.get("corrected_answer")
    chk(isinstance(ca, str) and ca.strip() != "", f"{sid}: empty corrected_answer")
    c = r.get("confidence")
    chk(isinstance(c, (int,float)) and 0.0 <= c <= 1.0, f"{sid}: confidence out of range")
    qd = r.get("quality_dimensions")
    chk(isinstance(qd, dict), f"{sid}: quality_dimensions not object")
    if isinstance(qd, dict):
        for dim in ("technical_correctness","instruction_coverage","operational_safety"):
            v = qd.get(dim)
            chk(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5,
                f"{sid}: quality_dimensions.{dim} invalid ({v!r})")
    chk(isinstance(r.get("risks"), list) and all(isinstance(x,str) for x in r.get("risks",[])),
        f"{sid}: risks not list[str]")
    chk(isinstance(r.get("evidence_required"), list) and all(isinstance(x,str) for x in r.get("evidence_required",[])),
        f"{sid}: evidence_required not list[str]")
    src = cmap.get(sid)
    chk(src is not None, f"{sid}: not found in corpus")
    if src:
        chk(r.get("source_user") == src[1], f"{sid}: source_user mismatch")
        chk(r.get("source_assistant") == src[2], f"{sid}: source_assistant mismatch")

# --- global aggregation across all batches ---
def batch_num(fn):
    return int(re.search(r"-(\d{4})\.jsonl$", fn).group(1))

tr_files = sorted([f for f in os.listdir(RES) if f.startswith("train-batch-")], key=batch_num)
va_files = sorted([f for f in os.listdir(RES) if f.startswith("validation-batch-")], key=batch_num)
chk(sorted(os.listdir(RES)) == sorted(tr_files + va_files), "unexpected files in results/")

def agg(files):
    ids = []
    for fn in files:
        txt = open(os.path.join(RES, fn), "rb").read().decode("utf-8")
        chk(txt.endswith("\n"), f"{fn}: no trailing newline")
        for j, l in enumerate(txt.split("\n"), 1):
            if l == "":
                continue
            try:
                ids.append(json.loads(l)["source_id"])
            except Exception as e:
                fails.append(f"{fn}:{j} parse error {e}")
    return ids

tr_ids = agg(tr_files)
va_ids = agg(va_files)
chk(len(set(tr_ids + va_ids)) == len(tr_ids) + len(va_ids), "duplicate source_id across batches")

# contiguous numbering check
chk([batch_num(f) for f in tr_files] == list(range(1, len(tr_files)+1)),
    "train batch numbering not contiguous from 0001")

exp_tr = [t[0] for t in train][:len(tr_ids)]
chk(tr_ids == exp_tr, "train aggregate is NOT a strict prefix of train.jsonl")
exp_va = [t[0] for t in val][:len(va_ids)]
chk(va_ids == exp_va, "validation aggregate is NOT a strict prefix of validation.jsonl")

from collections import Counter
dec = Counter(r["decision"] for r in recs)

print("train=%d/5399 validation=%d/601 total=%d/6000" % (len(tr_ids), len(va_ids), len(tr_ids)+len(va_ids)))
print("this_batch_ids=%s..%s" % (recs[0]["source_id"], recs[-1]["source_id"]))
print("decisions=%s" % dict(dec))
if fails:
    print("VERIFY_FAIL (%d)" % len(fails))
    for f in fails[:40]:
        print("  -", f)
    sys.exit(1)
print("VERIFY_PASS")
