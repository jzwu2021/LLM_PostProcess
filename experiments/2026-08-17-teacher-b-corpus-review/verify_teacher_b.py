import json, os, sys, re, glob

ROOT = "/media/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
errs = []

def load_corpus(name):
    out = []
    for ln in open(os.path.join(ROOT, "research/ai-infra-expert/corpus", name)):
        d = json.loads(ln)
        m = d["messages"]
        out.append((d["id"],
                    [x for x in m if x["role"] == "user"][0]["content"],
                    [x for x in m if x["role"] == "assistant"][0]["content"]))
    return out

corp = {"train": load_corpus("train.jsonl"), "validation": load_corpus("validation.jsonl")}
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]

seen = {}
seq = {"train": [], "validation": []}
for split in ("train","validation"):
    for path in sorted(glob.glob(os.path.join(RES, f"{split}-batch-*.jsonl"))):
        for i, ln in enumerate(open(path), 1):
            ln = ln.rstrip("\n")
            if not ln.strip():
                errs.append(f"{path}:{i} blank line"); continue
            try:
                r = json.loads(ln)
            except Exception as e:
                errs.append(f"{path}:{i} bad json {e}"); continue
            for k in REQ:
                if k not in r: errs.append(f"{path}:{i} missing {k}")
            if len(r) != len(REQ): errs.append(f"{path}:{i} field count {len(r)}")
            if r.get("teacher_lane") != "teacher-B": errs.append(f"{path}:{i} lane")
            if r.get("teacher_model") != "claude-opus-5-current": errs.append(f"{path}:{i} model")
            if r.get("calibration_status") != "provisional": errs.append(f"{path}:{i} status")
            if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{path}:{i} decision")
            if not isinstance(r.get("corrected_answer"), str) or not r["corrected_answer"].strip():
                errs.append(f"{path}:{i} empty corrected_answer")
            c = r.get("confidence")
            if not isinstance(c,(int,float)) or not (0 <= c <= 1): errs.append(f"{path}:{i} confidence")
            qd = r.get("quality_dimensions")
            if not isinstance(qd, dict): errs.append(f"{path}:{i} qd type")
            else:
                for k in ("technical_correctness","instruction_coverage","operational_safety"):
                    v = qd.get(k)
                    if not isinstance(v,int) or isinstance(v,bool) or not (1<=v<=5):
                        errs.append(f"{path}:{i} qd {k}")
            for k in ("risks","evidence_required"):
                v = r.get(k)
                if not isinstance(v,list) or not all(isinstance(x,str) for x in v):
                    errs.append(f"{path}:{i} {k} type")
            sid = r.get("source_id")
            if sid in seen: errs.append(f"{path}:{i} duplicate source_id {sid}")
            seen[sid] = path
            seq[split].append((sid, r.get("source_user"), r.get("source_assistant")))

for split in ("train","validation"):
    got, exp = seq[split], corp[split]
    if len(got) > len(exp):
        errs.append(f"{split}: more records ({len(got)}) than corpus ({len(exp)})")
    for i,(g,e) in enumerate(zip(got,exp)):
        if g[0] != e[0]: errs.append(f"{split}[{i}] id order {g[0]} != {e[0]}"); break
        if g[1] != e[1]: errs.append(f"{split}[{i}] {g[0]} source_user mismatch")
        if g[2] != e[2]: errs.append(f"{split}[{i}] {g[0]} source_assistant mismatch")

print(f"train={len(seq['train'])}/5399 validation={len(seq['validation'])}/601 total={len(seq['train'])+len(seq['validation'])}/6000")
if errs:
    print("VERIFY_FAIL", len(errs))
    for e in errs[:30]: print(" ", e)
    sys.exit(1)
print("VERIFY_PASS")
