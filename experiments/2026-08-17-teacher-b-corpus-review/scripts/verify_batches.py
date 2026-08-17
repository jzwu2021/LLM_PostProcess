import json, glob, os, sys

BASE = "/home/johnson/workspace/LLM_PostProcess"
RES = f"{BASE}/experiments/2026-08-17-teacher-b-corpus-review/results"
errs = []

def load_corpus(name):
    return [json.loads(l) for l in open(f"{BASE}/research/ai-infra-expert/corpus/{name}.jsonl")]

corp = {"train": load_corpus("train"), "validation": load_corpus("validation")}

REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]

seen = set()
agg = {"train": [], "validation": []}
for f in sorted(glob.glob(f"{RES}/*.jsonl")):
    split = "train" if os.path.basename(f).startswith("train") else "validation"
    for i, line in enumerate(open(f), 1):
        try:
            r = json.loads(line)
        except Exception as e:
            errs.append(f"{f}:{i} JSON parse: {e}"); continue
        for k in REQ:
            if k not in r: errs.append(f"{f}:{i} missing field {k}")
        if r.get("teacher_lane") != "teacher-B": errs.append(f"{f}:{i} bad lane")
        if r.get("teacher_model") != "claude-opus-5-current": errs.append(f"{f}:{i} bad model")
        if r.get("calibration_status") != "provisional": errs.append(f"{f}:{i} bad status")
        if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{f}:{i} bad decision")
        if not isinstance(r.get("corrected_answer"), str) or not r["corrected_answer"].strip():
            errs.append(f"{f}:{i} empty corrected_answer")
        c = r.get("confidence")
        if not isinstance(c,(int,float)) or not (0.0 <= c <= 1.0): errs.append(f"{f}:{i} bad confidence")
        qd = r.get("quality_dimensions")
        if not isinstance(qd, dict): errs.append(f"{f}:{i} qd not object")
        else:
            for d in ("technical_correctness","instruction_coverage","operational_safety"):
                v = qd.get(d)
                if not isinstance(v,int) or isinstance(v,bool) or not (1<=v<=5):
                    errs.append(f"{f}:{i} bad qd.{d}")
        for k in ("risks","evidence_required"):
            v = r.get(k)
            if not isinstance(v,list) or not all(isinstance(x,str) for x in v):
                errs.append(f"{f}:{i} {k} not list[str]")
        sid = r.get("source_id")
        if sid in seen: errs.append(f"{f}:{i} duplicate source_id {sid}")
        seen.add(sid)
        agg[split].append(r)

for split in ("train","validation"):
    rows = agg[split]
    src = corp[split]
    if len(rows) > len(src):
        errs.append(f"{split}: more rows ({len(rows)}) than corpus ({len(src)})")
    for idx, r in enumerate(rows):
        d = src[idx]
        if r["source_id"] != d["id"]:
            errs.append(f"{split}[{idx}] prefix order mismatch: {r['source_id']} != {d['id']}"); break
        u = [m["content"] for m in d["messages"] if m["role"]=="user"][0]
        a = [m["content"] for m in d["messages"] if m["role"]=="assistant"][0]
        if r["source_user"] != u: errs.append(f"{split}[{idx}] source_user mismatch")
        if r["source_assistant"] != a: errs.append(f"{split}[{idx}] source_assistant mismatch")

print(f"train={len(agg['train'])}/5399 validation={len(agg['validation'])}/601 total={len(agg['train'])+len(agg['validation'])}/6000")
if errs:
    print("VERIFY_FAIL", len(errs))
    for e in errs[:40]: print(" ", e)
    sys.exit(1)
print("VERIFY_PASS")
