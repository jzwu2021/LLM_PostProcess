import json, sys, os, glob, hashlib

EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
RES = os.path.join(EXP, "results")
CORP = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus"
BATCH = os.path.join(RES, "train-batch-0159.jsonl")
EXPECT_N = 10
fails = []


def ck(c, m):
    if not c:
        fails.append(m)


REQ = ["source_id", "teacher_lane", "teacher_model", "calibration_status", "decision",
       "source_user", "source_assistant", "corrected_answer", "quality_dimensions",
       "risks", "evidence_required", "confidence"]


def load_corpus(name):
    out = []
    for l in open(os.path.join(CORP, name), encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        u = [m["content"] for m in d["messages"] if m["role"] == "user"][0]
        a = [m["content"] for m in d["messages"] if m["role"] == "assistant"][0]
        out.append((d["id"], u, a))
    return out


train = load_corpus("train.jsonl")
val = load_corpus("validation.jsonl")
tmap = {i: (u, a) for i, u, a in train}

raw = open(BATCH, "rb").read()
ck(raw.endswith(b"\n"), "batch not newline terminated")
ck(b"\r" not in raw, "batch contains CR characters")
lines = raw.decode("utf-8").split("\n")[:-1]
ck(len(lines) == EXPECT_N, f"batch line count {len(lines)} != {EXPECT_N}")

recs = []
for n, l in enumerate(lines, 1):
    try:
        r = json.loads(l)
    except Exception as e:
        fails.append(f"line {n} unparseable: {e}")
        continue
    recs.append(r)
    for k in REQ:
        ck(k in r, f"line {n} missing field {k}")
    ck(len(set(r.keys()) - set(REQ)) == 0, f"line {n} has unexpected extra fields")
    ck(r.get("teacher_lane") == "teacher-B", f"line {n} bad teacher_lane")
    ck(r.get("teacher_model") == "claude-opus-5-current", f"line {n} bad teacher_model")
    ck(r.get("calibration_status") == "provisional", f"line {n} bad calibration_status")
    ck(r.get("decision") in ("keep", "rewrite", "reject"), f"line {n} bad decision")
    ca = r.get("corrected_answer")
    ck(isinstance(ca, str) and ca.strip() != "", f"line {n} empty corrected_answer")
    ck(isinstance(ca, str) and len(ca) > 400, f"line {n} corrected_answer suspiciously short")
    c = r.get("confidence")
    ck(isinstance(c, (int, float)) and not isinstance(c, bool) and 0.0 <= c <= 1.0,
       f"line {n} bad confidence {c}")
    qd = r.get("quality_dimensions")
    ck(isinstance(qd, dict), f"line {n} quality_dimensions not object")
    if isinstance(qd, dict):
        ck(set(qd.keys()) == {"technical_correctness", "instruction_coverage", "operational_safety"},
           f"line {n} quality_dimensions key set wrong")
        for d in ("technical_correctness", "instruction_coverage", "operational_safety"):
            v = qd.get(d)
            ck(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5,
               f"line {n} quality_dimensions.{d} bad: {v}")
    ck(isinstance(r.get("risks"), list) and len(r["risks"]) > 0
       and all(isinstance(x, str) and x.strip() for x in r["risks"]), f"line {n} risks bad")
    ck(isinstance(r.get("evidence_required"), list) and len(r["evidence_required"]) > 0
       and all(isinstance(x, str) and x.strip() for x in r["evidence_required"]),
       f"line {n} evidence_required bad")
    sid = r.get("source_id")
    src = tmap.get(sid)
    ck(src is not None, f"line {n} source_id {sid} not present in train corpus")
    if src:
        ck(r.get("source_user") == src[0], f"line {n} source_user not char-identical to corpus")
        ck(r.get("source_assistant") == src[1], f"line {n} source_assistant not char-identical to corpus")
        ck(r.get("corrected_answer") != src[1], f"line {n} corrected_answer identical to source_assistant")

# anti-template: distinct corrected answers within batch
digests = [hashlib.sha256(r["corrected_answer"].encode("utf-8")).hexdigest() for r in recs]
ck(len(set(digests)) == len(digests), "duplicate corrected_answer sha256 inside batch")

# global aggregation
def agg(prefix):
    ids = []
    for p in sorted(glob.glob(os.path.join(RES, prefix + "-batch-*.jsonl"))):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                ids.append(json.loads(l)["source_id"])
    return ids


tids = agg("train")
vids = agg("validation")
ck(len(set(tids)) == len(tids), "duplicate source_id in train aggregate")
ck(len(set(vids)) == len(vids), "duplicate source_id in validation aggregate")
ck(not (set(tids) & set(vids)), "train/validation source_id overlap")
ck(tids == [i for i, _, _ in train][:len(tids)], "train aggregate is not a strict corpus prefix")
ck(vids == [i for i, _, _ in val][:len(vids)], "validation aggregate is not a strict corpus prefix")

print(f"batch_records={len(recs)} train_total={len(tids)} validation_total={len(vids)} "
      f"total={len(tids)+len(vids)} corpus_train={len(train)} corpus_val={len(val)}")
if fails:
    print("VERIFY_FAIL")
    for f in fails[:40]:
        print(" -", f)
    sys.exit(1)
print("VERIFY_PASS")
