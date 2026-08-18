import json, os, sys, glob

EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
CORP = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus"
BATCH = os.path.join(EXP, "results", "train-batch-0150.jsonl")
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
fails = []

def load_corpus(name):
    out = []
    with open(os.path.join(CORP, name)) as f:
        for line in f:
            o = json.loads(line)
            m = {x["role"]: x["content"] for x in o["messages"]}
            out.append((o["id"], m["user"], m["assistant"]))
    return out

train = load_corpus("train.jsonl")
val = load_corpus("validation.jsonl")

# 1. batch parses line by line
raw = open(BATCH, "rb").read().decode()
lines = [l for l in raw.split("\n") if l.strip()]
if raw.count("\n") != len(lines):
    fails.append("newline separation anomaly")
batch = []
for i, l in enumerate(lines):
    try:
        batch.append(json.loads(l))
    except Exception as e:
        fails.append(f"line {i+1} not JSON: {e}")
if len(batch) != 10:
    fails.append(f"batch size {len(batch)} != 10")

for r in batch:
    for k in REQ:
        if k not in r:
            fails.append(f"{r.get('source_id')} missing {k}")
    if r.get("teacher_lane") != "teacher-B": fails.append(f"{r.get('source_id')} lane")
    if r.get("teacher_model") != "claude-opus-5-current": fails.append(f"{r.get('source_id')} model")
    if r.get("calibration_status") != "provisional": fails.append(f"{r.get('source_id')} status")
    if r.get("decision") not in ("keep","rewrite","reject"): fails.append(f"{r.get('source_id')} decision")
    if not isinstance(r.get("corrected_answer"), str) or not r["corrected_answer"].strip():
        fails.append(f"{r.get('source_id')} empty corrected_answer")
    c = r.get("confidence")
    if not isinstance(c,(int,float)) or not (0.0 <= c <= 1.0): fails.append(f"{r.get('source_id')} confidence")
    qd = r.get("quality_dimensions")
    if not isinstance(qd, dict): fails.append(f"{r.get('source_id')} qd type")
    else:
        for d in ("technical_correctness","instruction_coverage","operational_safety"):
            v = qd.get(d)
            if not isinstance(v,int) or isinstance(v,bool) or not (1 <= v <= 5):
                fails.append(f"{r.get('source_id')} qd.{d}")
    for arr in ("risks","evidence_required"):
        v = r.get(arr)
        if not isinstance(v,list) or not all(isinstance(x,str) for x in v):
            fails.append(f"{r.get('source_id')} {arr} type")

# 2. aggregate all batches, global uniqueness + prefix
def agg(prefix):
    files = sorted(glob.glob(os.path.join(EXP,"results",prefix+"-batch-*.jsonl")))
    recs = []
    for fp in files:
        for l in open(fp):
            if l.strip():
                recs.append((os.path.basename(fp), json.loads(l)))
    return files, recs

tf, trecs = agg("train")
vf, vrecs = agg("validation")
allrecs = trecs + vrecs
ids = [r["source_id"] for _, r in allrecs]
if len(ids) != len(set(ids)):
    dup = [x for x in set(ids) if ids.count(x) > 1]
    fails.append(f"duplicate source_id: {dup[:5]}")

def check_prefix(recs, corpus, label):
    if len(recs) > len(corpus):
        fails.append(f"{label} overrun {len(recs)}>{len(corpus)}")
        return
    for i, (fn, r) in enumerate(recs):
        cid, cu, ca = corpus[i]
        if r["source_id"] != cid:
            fails.append(f"{label} idx {i} id {r['source_id']} != {cid} ({fn})"); return
        if r["source_user"] != cu:
            fails.append(f"{label} idx {i} source_user mismatch ({fn})")
        if r["source_assistant"] != ca:
            fails.append(f"{label} idx {i} source_assistant mismatch ({fn})")

check_prefix(trecs, train, "train")
check_prefix(vrecs, val, "validation")

print("train_files", len(tf), "train_records", len(trecs))
print("validation_files", len(vf), "validation_records", len(vrecs))
print("total", len(allrecs))
print("corpus sizes", len(train), len(val))
if fails:
    print("VERIFY_FAIL")
    for f in fails[:40]: print(" -", f)
    sys.exit(1)
print("VERIFY_PASS")
