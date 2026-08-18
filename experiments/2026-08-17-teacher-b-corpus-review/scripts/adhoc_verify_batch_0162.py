import json, os, glob, re, sys

EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
CORP = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus"
BATCH = os.path.join(EXP, "results", "train-batch-0162.jsonl")
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
fail = []

def ck(c, msg):
    if not c: fail.append(msg)

raw = open(BATCH, encoding="utf-8").read()
lines = raw.split("\n")
ck(lines[-1] == "", "batch file must end with newline")
lines = [l for l in lines if l != ""]
ck(len(lines) == 10, f"batch line count {len(lines)} != 10")
recs = []
for i, l in enumerate(lines):
    try:
        recs.append(json.loads(l))
    except Exception as e:
        fail.append(f"line {i+1} not valid JSON: {e}")

corp = {}
order = {}
for split in ("train", "validation"):
    rows = [json.loads(l) for l in open(os.path.join(CORP, split + ".jsonl"), encoding="utf-8")]
    order[split] = [r["id"] for r in rows]
    for r in rows:
        u = [x["content"] for x in r["messages"] if x["role"] == "user"][0]
        a = [x["content"] for x in r["messages"] if x["role"] == "assistant"][0]
        corp[r["id"]] = (u, a)

for r in recs:
    sid = r.get("source_id")
    for k in REQ:
        ck(k in r, f"{sid}: missing field {k}")
    ck(r.get("teacher_lane") == "teacher-B", f"{sid}: bad teacher_lane")
    ck(r.get("teacher_model") == "claude-opus-5-current", f"{sid}: bad teacher_model")
    ck(r.get("calibration_status") == "provisional", f"{sid}: bad calibration_status")
    ck(r.get("decision") in ("keep","rewrite","reject"), f"{sid}: bad decision")
    ck(sid in corp, f"{sid}: not in corpus")
    if sid in corp:
        ck(r["source_user"] == corp[sid][0], f"{sid}: source_user mismatch")
        ck(r["source_assistant"] == corp[sid][1], f"{sid}: source_assistant mismatch")
    ck(isinstance(r.get("corrected_answer"), str) and r["corrected_answer"].strip() != "", f"{sid}: empty corrected_answer")
    qd = r.get("quality_dimensions")
    ck(isinstance(qd, dict), f"{sid}: quality_dimensions not object")
    if isinstance(qd, dict):
        for d in ("technical_correctness","instruction_coverage","operational_safety"):
            v = qd.get(d)
            ck(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5, f"{sid}: bad {d}={v}")
    ck(isinstance(r.get("risks"), list) and all(isinstance(x,str) for x in r["risks"]), f"{sid}: risks not str list")
    ck(isinstance(r.get("evidence_required"), list) and all(isinstance(x,str) for x in r["evidence_required"]), f"{sid}: evidence_required not str list")
    c = r.get("confidence")
    ck(isinstance(c, float) and 0.0 <= c <= 1.0, f"{sid}: confidence {c} out of range")

# global aggregate: uniqueness + prefix
for split in ("train","validation"):
    ids = []
    files = sorted(glob.glob(os.path.join(EXP, "results", f"{split}-batch-*.jsonl")))
    nums = [int(re.search(r"-(\d{4})\.jsonl$", f).group(1)) for f in files]
    ck(nums == sorted(nums) and (not nums or nums == list(range(nums[0], nums[0]+len(nums)))), f"{split}: batch numbering not contiguous {nums[:3]}..{nums[-3:] if nums else []}")
    for f in files:
        for l in open(f, encoding="utf-8"):
            if l.strip():
                ids.append(json.loads(l)["source_id"])
    ck(len(ids) == len(set(ids)), f"{split}: duplicate source_id")
    ck(ids == order[split][:len(ids)], f"{split}: not a strict prefix of corpus order")
    print(f"{split}: {len(ids)} records, prefix_ok={ids == order[split][:len(ids)]}")

allids = []
for f in glob.glob(os.path.join(EXP, "results", "*.jsonl")):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            allids.append(json.loads(l)["source_id"])
ck(len(allids) == len(set(allids)), "global duplicate source_id across all results")
print("TOTAL", len(allids))

if fail:
    print("VERIFY_FAIL")
    for f_ in fail[:40]: print(" -", f_)
    sys.exit(1)
print("VERIFY_PASS")
