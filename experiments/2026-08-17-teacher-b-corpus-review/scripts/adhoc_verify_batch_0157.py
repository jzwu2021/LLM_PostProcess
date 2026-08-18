"""Independent ad-hoc verifier for teacher-B batch 0157.
Written fresh for this batch; does not import the generator.
Re-derives everything from the raw corpus and the raw result files.
"""
import json, glob, os, sys, hashlib, re

ROOT = "experiments/2026-08-17-teacher-b-corpus-review/results"
BATCH = os.path.join(ROOT, "train-batch-0157.jsonl")
TRAIN = "research/ai-infra-expert/corpus/train.jsonl"
VAL = "research/ai-infra-expert/corpus/validation.jsonl"
EXPECT_N = 10
REQUIRED = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
            "source_user","source_assistant","corrected_answer","quality_dimensions",
            "risks","evidence_required","confidence"]

fails = []
def chk(cond, msg):
    if not cond:
        fails.append(msg)

def corpus_pairs(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = r["messages"]
        u = next(x["content"] for x in m if x["role"] == "user")
        a = next(x["content"] for x in m if x["role"] == "assistant")
        out.append((r["id"], u, a))
    return out

train = corpus_pairs(TRAIN)
val = corpus_pairs(VAL)
tmap = {i: (u, a) for i, u, a in train}

# --- 1. batch parses line by line, physical newline separated
raw = open(BATCH, "rb").read()
chk(raw.endswith(b"\n"), "batch does not end with newline")
lines = raw.decode("utf-8").split("\n")
if lines and lines[-1] == "":
    lines = lines[:-1]
chk(len(lines) == EXPECT_N, f"batch line count {len(lines)} != {EXPECT_N}")
recs = []
for i, ln in enumerate(lines):
    try:
        recs.append(json.loads(ln))
    except Exception as e:
        fails.append(f"line {i+1} not valid JSON: {e}")
chk(b"\r" not in raw, "batch contains CR characters")

# --- 2. per record schema
for i, r in enumerate(recs):
    tag = f"rec{i+1}({r.get('source_id')})"
    for f in REQUIRED:
        chk(f in r, f"{tag} missing field {f}")
    chk(set(r.keys()) == set(REQUIRED), f"{tag} unexpected field set: {sorted(set(r.keys())^set(REQUIRED))}")
    chk(r.get("teacher_lane") == "teacher-B", f"{tag} bad teacher_lane")
    chk(r.get("teacher_model") == "claude-opus-5-current", f"{tag} bad teacher_model")
    chk(r.get("calibration_status") == "provisional", f"{tag} bad calibration_status")
    chk(r.get("decision") in ("keep","rewrite","reject"), f"{tag} bad decision")
    ca = r.get("corrected_answer")
    chk(isinstance(ca, str) and ca.strip() != "", f"{tag} empty corrected_answer")
    c = r.get("confidence")
    chk(isinstance(c, float) and 0.0 <= c <= 1.0, f"{tag} confidence out of range/type")
    qd = r.get("quality_dimensions")
    chk(isinstance(qd, dict), f"{tag} quality_dimensions not object")
    if isinstance(qd, dict):
        chk(set(qd.keys()) == {"technical_correctness","instruction_coverage","operational_safety"},
            f"{tag} quality_dimensions keys wrong")
        for k, v in qd.items():
            chk(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5,
                f"{tag} quality_dimensions.{k} not int 1-5")
    for fld in ("risks","evidence_required"):
        v = r.get(fld)
        chk(isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) and x.strip() for x in v),
            f"{tag} {fld} not non-empty list of non-empty strings")
    # exact corpus equality
    sid = r.get("source_id")
    chk(sid in tmap, f"{tag} source_id not in train corpus")
    if sid in tmap:
        u, a = tmap[sid]
        chk(r.get("source_user") == u, f"{tag} source_user mismatch vs corpus")
        chk(r.get("source_assistant") == a, f"{tag} source_assistant mismatch vs corpus")

# --- 3. anti-template: corrected answers distinct within batch
hs = [hashlib.sha256(r["corrected_answer"].encode()).hexdigest() for r in recs if "corrected_answer" in r]
chk(len(set(hs)) == len(hs), "duplicate corrected_answer sha256 within batch")
chk(all(len(r.get("corrected_answer","")) > 800 for r in recs), "some corrected_answer suspiciously short")

# --- 4. aggregate: global uniqueness + strict corpus prefix
def agg(prefix):
    files = sorted(glob.glob(os.path.join(ROOT, prefix + "-batch-*.jsonl")))
    ids = []
    for fp in files:
        n = int(re.search(r"batch-(\d+)", fp).group(1))
        for ln in open(fp, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                ids.append(json.loads(ln)["source_id"])
    return files, ids

tfiles, tids = agg("train")
vfiles, vids = agg("validation")

# contiguous batch numbering
nums = [int(re.search(r"batch-(\d+)", f).group(1)) for f in tfiles]
chk(nums == list(range(1, len(nums)+1)), f"train batch numbering not contiguous from 0001: {nums[:3]}...{nums[-3:]}")

allids = tids + vids
chk(len(set(allids)) == len(allids), "duplicate source_id across all batches")

chk(tids == [i for i, _, _ in train][:len(tids)], "train aggregate is NOT a strict positional prefix of train corpus")
chk(vids == [i for i, _, _ in val][:len(vids)], "validation aggregate is NOT a strict positional prefix of validation corpus")
chk(len(tids) <= 5399, "train count exceeds 5399")
chk(len(vids) <= 601, "validation count exceeds 601")
chk(len(vids) == 0 or len(tids) == 5399, "validation started before train finished")

print(f"train={len(tids)}/5399 validation={len(vids)}/601 total={len(tids)+len(vids)}/6000 remaining={6000-len(tids)-len(vids)}")
print("batch decisions:", {d: sum(1 for r in recs if r.get('decision')==d) for d in ('keep','rewrite','reject')})
if fails:
    print("VERIFY_FAIL")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("VERIFY_PASS")
