import json, glob, re, sys

BASE = "experiments/2026-08-17-teacher-b-corpus-review/results"
BATCH = BASE + "/train-batch-0154.jsonl"
fails = []

def ck(c, m):
    if not c: fails.append(m)

# 1. batch parses line by line
raw = open(BATCH, encoding="utf-8").read()
ck(raw.endswith("\n"), "batch does not end with newline")
lines = raw.rstrip("\n").split("\n")
ck(len(lines) == 10, f"batch count {len(lines)} != 10")
recs = []
for i, l in enumerate(lines):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fails.append(f"line {i+1} unparseable: {e}")

REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]

corpus = {}
order = []
for l in open("research/ai-infra-expert/corpus/train.jsonl", encoding="utf-8"):
    d = json.loads(l)
    order.append(d["id"])
    msgs = d["messages"]
    corpus[d["id"]] = (
        [m["content"] for m in msgs if m["role"]=="user"][0],
        [m["content"] for m in msgs if m["role"]=="assistant"][0],
    )

for r in recs:
    sid = r.get("source_id","?")
    for f in REQ:
        ck(f in r, f"{sid}: missing field {f}")
    ck(set(r.keys()) == set(REQ), f"{sid}: unexpected fields {set(r.keys())-set(REQ)}")
    ck(r.get("teacher_lane")=="teacher-B", f"{sid}: bad teacher_lane")
    ck(r.get("teacher_model")=="claude-opus-5-current", f"{sid}: bad teacher_model")
    ck(r.get("calibration_status")=="provisional", f"{sid}: bad calibration_status")
    ck(r.get("decision") in ("keep","rewrite","reject"), f"{sid}: bad decision")
    ck(isinstance(r.get("corrected_answer"),str) and r["corrected_answer"].strip()!="", f"{sid}: empty corrected_answer")
    qd = r.get("quality_dimensions")
    ck(isinstance(qd,dict) and set(qd)=={"technical_correctness","instruction_coverage","operational_safety"}, f"{sid}: bad qd keys")
    if isinstance(qd,dict):
        for k,v in qd.items():
            ck(isinstance(v,int) and 1<=v<=5, f"{sid}: qd {k} bad")
    ck(isinstance(r.get("risks"),list) and all(isinstance(x,str) for x in r["risks"]), f"{sid}: bad risks")
    ck(isinstance(r.get("evidence_required"),list) and all(isinstance(x,str) for x in r["evidence_required"]), f"{sid}: bad evidence_required")
    c = r.get("confidence")
    ck(isinstance(c,(int,float)) and not isinstance(c,bool) and 0.0<=c<=1.0, f"{sid}: bad confidence")
    ck(sid in corpus, f"{sid}: not in corpus")
    if sid in corpus:
        cu, ca = corpus[sid]
        ck(r.get("source_user")==cu, f"{sid}: source_user mismatch")
        ck(r.get("source_assistant")==ca, f"{sid}: source_assistant mismatch")

# global uniqueness + prefix
def nk(p):
    return int(re.search(r"-(\d{4})\.jsonl$", p).group(1))

allids = []
for p in sorted(glob.glob(BASE+"/train-batch-*.jsonl"), key=nk):
    for l in open(p, encoding="utf-8"):
        l=l.strip()
        if l: allids.append(json.loads(l)["source_id"])
ck(len(allids)==len(set(allids)), "duplicate source_id across train batches")
ck(allids == order[:len(allids)], "aggregated train sequence is not a strict corpus prefix")

valids = []
for p in sorted(glob.glob(BASE+"/validation-batch-*.jsonl"), key=nk):
    for l in open(p, encoding="utf-8"):
        l=l.strip()
        if l: valids.append(json.loads(l)["source_id"])
ck(not set(allids)&set(valids), "id overlap train/validation")

print("TRAIN_TOTAL", len(allids), "VAL_TOTAL", len(valids))
if fails:
    print("VERIFY_FAIL")
    for f in fails[:40]: print(" -", f)
    sys.exit(1)
print("VERIFY_PASS")
