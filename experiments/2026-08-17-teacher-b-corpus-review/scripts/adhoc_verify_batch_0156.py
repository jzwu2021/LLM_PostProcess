import json, glob, os, re, sys

ROOT = "experiments/2026-08-17-teacher-b-corpus-review/results"
BATCH = os.path.join(ROOT, "train-batch-0156.jsonl")
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
fails = []

raw = open(BATCH, encoding="utf-8").read()
lines = raw.split("\n")
assert lines[-1] == "", "file must end with newline"
lines = lines[:-1]
if len(lines) != 10: fails.append("batch line count %d" % len(lines))
recs = []
for i, l in enumerate(lines):
    try: recs.append(json.loads(l))
    except Exception as e: fails.append("line %d parse: %s" % (i, e))

def load_corpus(p):
    out = []
    for l in open(p, encoding="utf-8"):
        d = json.loads(l)
        u = [m["content"] for m in d["messages"] if m["role"] == "user"][0]
        a = [m["content"] for m in d["messages"] if m["role"] == "assistant"][0]
        out.append((d["id"], u, a))
    return out

train = load_corpus("research/ai-infra-expert/corpus/train.jsonl")
val = load_corpus("research/ai-infra-expert/corpus/validation.jsonl")
tmap = {x[0]: x for x in train}
vmap = {x[0]: x for x in val}

for r in recs:
    sid = r.get("source_id")
    for k in REQ:
        if k not in r: fails.append("%s missing %s" % (sid, k))
    if r.get("teacher_lane") != "teacher-B": fails.append("%s lane" % sid)
    if r.get("teacher_model") != "claude-opus-5-current": fails.append("%s model" % sid)
    if r.get("calibration_status") != "provisional": fails.append("%s status" % sid)
    if r.get("decision") not in ("keep","rewrite","reject"): fails.append("%s decision" % sid)
    if not isinstance(r.get("corrected_answer"), str) or not r["corrected_answer"].strip():
        fails.append("%s empty corrected_answer" % sid)
    c = r.get("confidence")
    if not isinstance(c,(int,float)) or not (0.0 <= c <= 1.0): fails.append("%s confidence" % sid)
    qd = r.get("quality_dimensions")
    if not isinstance(qd, dict): fails.append("%s qd type" % sid)
    else:
        for k in ("technical_correctness","instruction_coverage","operational_safety"):
            v = qd.get(k)
            if not isinstance(v,int) or isinstance(v,bool) or not (1<=v<=5): fails.append("%s qd %s" % (sid,k))
    for k in ("risks","evidence_required"):
        v = r.get(k)
        if not isinstance(v,list) or not all(isinstance(x,str) for x in v): fails.append("%s %s type" % (sid,k))
    src = tmap.get(sid) or vmap.get(sid)
    if not src: fails.append("%s not in corpus" % sid)
    else:
        if r.get("source_user") != src[1]: fails.append("%s source_user mismatch" % sid)
        if r.get("source_assistant") != src[2]: fails.append("%s source_assistant mismatch" % sid)

# global aggregation
def agg(prefix):
    files = sorted(glob.glob(os.path.join(ROOT, prefix + "-batch-*.jsonl")))
    ids = []
    for f in files:
        for l in open(f, encoding="utf-8"):
            l = l.strip()
            if l: ids.append(json.loads(l)["source_id"])
    return ids

tids = agg("train"); vids = agg("validation")
allids = tids + vids
if len(allids) != len(set(allids)): fails.append("duplicate source_id globally")
if tids != [x[0] for x in train[:len(tids)]]: fails.append("train not strict prefix")
if vids != [x[0] for x in val[:len(vids)]]: fails.append("validation not strict prefix")

print("train_processed", len(tids), "validation_processed", len(vids), "total", len(allids))
print("FAILS:", len(fails))
for f in fails[:40]: print("  ", f)
sys.exit(1 if fails else 0)
