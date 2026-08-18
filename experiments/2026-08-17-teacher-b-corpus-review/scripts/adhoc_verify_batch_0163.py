import json, os, glob, sys

EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
CORP = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
BATCH = os.path.join(EXP, "results", "train-batch-0163.jsonl")
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user",
       "source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]
fail = []
def ck(c, m):
    if not c: fail.append(m)

raw = open(BATCH, encoding="utf-8").read()
ck(raw.endswith("\n"), "batch not newline-terminated")
lines = [l for l in raw.split("\n") if l != ""]
ck(len(lines) == 10, f"expected 10 lines got {len(lines)}")
recs = []
for i, l in enumerate(lines):
    try: recs.append(json.loads(l))
    except Exception as e: fail.append(f"line {i+1} unparseable: {e}")

corpus = [json.loads(l) for l in open(CORP, encoding="utf-8")]
by_id = {r["id"]: r for r in corpus}

for r in recs:
    sid = r.get("source_id")
    for k in REQ: ck(k in r, f"{sid} missing {k}")
    ck(r.get("teacher_lane") == "teacher-B", f"{sid} bad lane")
    ck(r.get("teacher_model") == "claude-opus-5-current", f"{sid} bad model")
    ck(r.get("calibration_status") == "provisional", f"{sid} bad status")
    ck(r.get("decision") in ("keep","rewrite","reject"), f"{sid} bad decision")
    src = by_id.get(sid)
    ck(src is not None, f"{sid} not in corpus")
    if src:
        su = [m for m in src["messages"] if m["role"]=="user"][0]["content"]
        sa = [m for m in src["messages"] if m["role"]=="assistant"][0]["content"]
        ck(r["source_user"] == su, f"{sid} source_user mismatch")
        ck(r["source_assistant"] == sa, f"{sid} source_assistant mismatch")
    ck(isinstance(r.get("corrected_answer"), str) and r["corrected_answer"].strip(), f"{sid} empty corrected_answer")
    qd = r.get("quality_dimensions", {})
    for k in ("technical_correctness","instruction_coverage","operational_safety"):
        v = qd.get(k)
        ck(isinstance(v, int) and 1 <= v <= 5, f"{sid} bad qd {k}")
    ck(isinstance(r.get("risks"), list) and all(isinstance(x,str) for x in r["risks"]), f"{sid} bad risks")
    ck(isinstance(r.get("evidence_required"), list) and all(isinstance(x,str) for x in r["evidence_required"]), f"{sid} bad evidence")
    c = r.get("confidence")
    ck(isinstance(c, float) and 0.0 <= c <= 1.0, f"{sid} bad confidence")

# global aggregate
files = sorted(glob.glob(os.path.join(EXP, "results", "train-batch-*.jsonl")))
allrecs = []
for fp in files:
    for l in open(fp, encoding="utf-8"):
        if l.strip(): allrecs.append(json.loads(l))
ids = [r["source_id"] for r in allrecs]
ck(len(ids) == len(set(ids)), "duplicate source_id globally")
expect = [r["id"] for r in corpus[:len(ids)]]
ck(ids == expect, "aggregated train sequence is not a strict prefix of train.jsonl")
nums = sorted(int(os.path.basename(f)[12:16]) for f in files)
ck(nums == list(range(1, len(files)+1)), "batch numbering not contiguous")
ck(len(glob.glob(os.path.join(EXP,"results","validation-batch-*"))) == 0, "validation files present")

print("total_records", len(allrecs))
if fail:
    print("FAIL"); [print(" -", m) for m in fail]; sys.exit(1)
print("PASS")
