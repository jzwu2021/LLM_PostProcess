import json, os, re, sys, glob

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
RES = os.path.join(EXP, "results")
BATCH = os.path.join(RES, "train-batch-0153.jsonl")
TRAIN = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
VAL = os.path.join(ROOT, "research/ai-infra-expert/corpus/validation.jsonl")

fails = []
def chk(cond, msg):
    if not cond:
        fails.append(msg)

REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]

# --- 1. raw JSONL parse, physical newline separated
raw = open(BATCH, "rb").read().decode()
lines = raw.split("\n")
chk(lines[-1] == "", "batch file must end with newline")
lines = [l for l in lines if l != ""]
chk(len(lines) == 10, f"expected 10 lines, got {len(lines)}")
recs = []
for i, l in enumerate(lines):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fails.append(f"line {i+1} not valid JSON: {e}")

# --- 2. corpus source of truth
def load(p):
    out = []
    for l in open(p):
        d = json.loads(l)
        m = d["messages"]
        out.append((d["id"],
                    [x for x in m if x["role"]=="user"][0]["content"],
                    [x for x in m if x["role"]=="assistant"][0]["content"]))
    return out
train = load(TRAIN)
val = load(VAL)
chk(len(train) == 5399, f"train corpus len {len(train)}")
chk(len(val) == 601, f"val corpus len {len(val)}")

tmap = {t[0]: t for t in train}

# --- 3. per-record schema
for i, r in enumerate(recs):
    tag = f"rec{i+1}({r.get('source_id')})"
    for k in REQ:
        chk(k in r, f"{tag} missing field {k}")
    chk(set(r.keys()) >= set(REQ), f"{tag} field set")
    chk(r.get("teacher_lane") == "teacher-B", f"{tag} teacher_lane")
    chk(r.get("teacher_model") == "claude-opus-5-current", f"{tag} teacher_model")
    chk(r.get("calibration_status") == "provisional", f"{tag} calibration_status")
    chk(r.get("decision") in ("keep","rewrite","reject"), f"{tag} decision")
    ca = r.get("corrected_answer")
    chk(isinstance(ca, str) and ca.strip() != "", f"{tag} corrected_answer empty")
    qd = r.get("quality_dimensions")
    chk(isinstance(qd, dict), f"{tag} qd not dict")
    if isinstance(qd, dict):
        for dim in ("technical_correctness","instruction_coverage","operational_safety"):
            v = qd.get(dim)
            chk(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5,
                f"{tag} qd.{dim}={v!r}")
    chk(isinstance(r.get("risks"), list) and all(isinstance(x,str) for x in r.get("risks",[])),
        f"{tag} risks")
    chk(isinstance(r.get("evidence_required"), list) and all(isinstance(x,str) for x in r.get("evidence_required",[])),
        f"{tag} evidence_required")
    c = r.get("confidence")
    chk(isinstance(c,(int,float)) and not isinstance(c,bool) and 0.0 <= c <= 1.0, f"{tag} confidence={c!r}")
    # exact corpus match
    src = tmap.get(r.get("source_id"))
    chk(src is not None, f"{tag} source_id not in train corpus")
    if src:
        chk(r.get("source_user") == src[1], f"{tag} source_user mismatch")
        chk(r.get("source_assistant") == src[2], f"{tag} source_assistant mismatch")

# --- 4. aggregate: global uniqueness + strict prefix
def agg(prefix):
    files = sorted(glob.glob(os.path.join(RES, prefix + "-batch-*.jsonl")))
    ids = []
    for fp in files:
        n = 0
        for l in open(fp):
            if l.strip() == "": continue
            ids.append(json.loads(l)["source_id"]); n += 1
        chk(n == 10 or fp == files[-1], f"{os.path.basename(fp)} has {n} records (expected 10)")
    # numbering contiguity
    nums = [int(re.search(r"-batch-(\d{4})\.jsonl$", f).group(1)) for f in files]
    chk(nums == list(range(1, len(files)+1)), f"{prefix} batch numbering not contiguous from 0001")
    return ids

tids = agg("train")
vids = agg("validation")
allids = tids + vids
chk(len(allids) == len(set(allids)), "duplicate source_id across all batches")

chk(tids == [t[0] for t in train[:len(tids)]], "train aggregate is NOT a strict prefix of train corpus")
chk(vids == [t[0] for t in val[:len(vids)]], "validation aggregate is NOT a strict prefix of validation corpus")

print(f"train_processed={len(tids)}/5399")
print(f"validation_processed={len(vids)}/601")
print(f"total={len(tids)+len(vids)}/6000")
print("batch_ids=", tids[-10:])
if fails:
    print("VERIFY=FAIL")
    for f in fails: print("  -", f)
    sys.exit(1)
print("VERIFY=PASS")
