import json, glob, os, sys, re

ROOT = "/media/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
errs = []

def load_corpus(name):
    ids, um, am = [], {}, {}
    with open(os.path.join(ROOT, "research/ai-infra-expert/corpus", name)) as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["id"])
            m = {x["role"]: x["content"] for x in d["messages"]}
            um[d["id"]] = m["user"]; am[d["id"]] = m["assistant"]
    return ids, um, am

REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]

all_ids = []
for split, fname in (("train","train.jsonl"), ("validation","validation.jsonl")):
    cids, um, am = load_corpus(fname)
    seq = []
    files = sorted(glob.glob(os.path.join(RES, f"{split}-batch-*.jsonl")))
    nums = [int(re.search(r"-(\d{4})\.jsonl$", p).group(1)) for p in files]
    if nums and nums != list(range(1, len(nums)+1)):
        errs.append(f"{split}: batch numbering not contiguous from 0001")
    for p in files:
        with open(p) as f:
            lines = f.read().split("\n")
        if lines and lines[-1] == "": lines.pop()
        if len(lines) != 10:
            errs.append(f"{p}: expected 10 lines got {len(lines)}")
        for i, ln in enumerate(lines, 1):
            try:
                r = json.loads(ln)
            except Exception as e:
                errs.append(f"{p}:{i} JSON parse fail {e}"); continue
            for k in REQ:
                if k not in r: errs.append(f"{p}:{i} missing {k}")
            if r.get("teacher_lane") != "teacher-B": errs.append(f"{p}:{i} bad lane")
            if r.get("teacher_model") != "claude-opus-5-current": errs.append(f"{p}:{i} bad model")
            if r.get("calibration_status") != "provisional": errs.append(f"{p}:{i} bad status")
            if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{p}:{i} bad decision")
            sid = r.get("source_id"); seq.append(sid); all_ids.append(sid)
            if sid not in um: errs.append(f"{p}:{i} unknown id {sid}")
            else:
                if r.get("source_user") != um[sid]: errs.append(f"{p}:{i} source_user mismatch {sid}")
                if r.get("source_assistant") != am[sid]: errs.append(f"{p}:{i} source_assistant mismatch {sid}")
            if not isinstance(r.get("corrected_answer"), str) or not r["corrected_answer"].strip():
                errs.append(f"{p}:{i} empty corrected_answer")
            qd = r.get("quality_dimensions")
            if not isinstance(qd, dict): errs.append(f"{p}:{i} qd not object")
            else:
                for k in ("technical_correctness","instruction_coverage","operational_safety"):
                    v = qd.get(k)
                    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 5:
                        errs.append(f"{p}:{i} qd.{k} invalid")
            for k in ("risks","evidence_required"):
                v = r.get(k)
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    errs.append(f"{p}:{i} {k} not string array")
            c = r.get("confidence")
            if not isinstance(c, (int, float)) or isinstance(c, bool) or not 0.0 <= c <= 1.0:
                errs.append(f"{p}:{i} confidence out of range")
    if seq != cids[:len(seq)]:
        errs.append(f"{split}: sequence is not a strict prefix of corpus order")
    print(f"{split}: {len(seq)} records")

if len(all_ids) != len(set(all_ids)):
    errs.append("duplicate source_id across batches")

print("TOTAL", len(all_ids))
if errs:
    print("VERIFY=FAIL")
    for e in errs[:50]: print(" ", e)
    sys.exit(1)
print("VERIFY=PASS")
