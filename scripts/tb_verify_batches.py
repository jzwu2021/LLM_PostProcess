import json, glob, os, re, sys

BASE = "/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(BASE, "experiments/2026-08-17-teacher-b-corpus-review/results")
CORPUS = {
    "train": os.path.join(BASE, "research/ai-infra-expert/corpus/train.jsonl"),
    "validation": os.path.join(BASE, "research/ai-infra-expert/corpus/validation.jsonl"),
}
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
THIS = sys.argv[1] if len(sys.argv) > 1 else None
errs = []

def load_corpus(p):
    seq = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            m = {x["role"]: x["content"] for x in r["messages"]}
            seq.append((r["id"], m.get("user",""), m.get("assistant","")))
    return seq

corp = {k: load_corpus(v) for k, v in CORPUS.items()}
seen = {}
agg = {"train": [], "validation": []}

for split in ("train","validation"):
    files = sorted(glob.glob(os.path.join(RES, f"{split}-batch-*.jsonl")))
    nums = [int(re.search(r"-(\d{4})\.jsonl$", f).group(1)) for f in files]
    if nums and nums != list(range(1, len(nums)+1)):
        errs.append(f"{split}: batch numbering not contiguous from 0001")
    for fp in files:
        raw = open(fp, "rb").read().decode("utf-8")
        if raw and not raw.endswith("\n"):
            errs.append(f"{fp}: no trailing newline")
        lines = [l for l in raw.split("\n") if l != ""]
        if THIS and os.path.basename(fp) == THIS and len(lines) != 10:
            errs.append(f"{fp}: expected 10 records, got {len(lines)}")
        for ln, l in enumerate(lines, 1):
            try:
                r = json.loads(l)
            except Exception as e:
                errs.append(f"{fp}:{ln} JSON parse: {e}"); continue
            for k in REQ:
                if k not in r: errs.append(f"{fp}:{ln} missing field {k}")
            if len(r) != len(REQ): errs.append(f"{fp}:{ln} unexpected field count {len(r)}")
            if r.get("teacher_lane") != "teacher-B": errs.append(f"{fp}:{ln} bad teacher_lane")
            if r.get("teacher_model") != "claude-opus-5-current": errs.append(f"{fp}:{ln} bad teacher_model")
            if r.get("calibration_status") != "provisional": errs.append(f"{fp}:{ln} bad calibration_status")
            if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{fp}:{ln} bad decision")
            ca = r.get("corrected_answer")
            if not isinstance(ca, str) or not ca.strip(): errs.append(f"{fp}:{ln} empty corrected_answer")
            c = r.get("confidence")
            if not isinstance(c,(int,float)) or isinstance(c,bool) or not (0.0 <= c <= 1.0):
                errs.append(f"{fp}:{ln} bad confidence {c}")
            qd = r.get("quality_dimensions")
            if not isinstance(qd, dict) or set(qd) != {"technical_correctness","instruction_coverage","operational_safety"}:
                errs.append(f"{fp}:{ln} bad quality_dimensions keys")
            else:
                for k,v in qd.items():
                    if not isinstance(v,int) or isinstance(v,bool) or not (1 <= v <= 5):
                        errs.append(f"{fp}:{ln} qd {k}={v} out of range")
            for k in ("risks","evidence_required"):
                v = r.get(k)
                if not isinstance(v, list) or not all(isinstance(x,str) for x in v):
                    errs.append(f"{fp}:{ln} {k} not list[str]")
            sid = r.get("source_id")
            if sid in seen: errs.append(f"{fp}:{ln} duplicate source_id {sid} (also {seen[sid]})")
            else: seen[sid] = f"{fp}:{ln}"
            agg[split].append((sid, r.get("source_user"), r.get("source_assistant")))

for split in ("train","validation"):
    a = agg[split]; c = corp[split]
    if len(a) > len(c):
        errs.append(f"{split}: more records ({len(a)}) than corpus ({len(c)})")
    for i,(got,exp) in enumerate(zip(a,c)):
        if got[0] != exp[0]: errs.append(f"{split}[{i}] id mismatch {got[0]} != {exp[0]}"); break
        if got[1] != exp[1]: errs.append(f"{split}[{i}] source_user mismatch for {exp[0]}")
        if got[2] != exp[2]: errs.append(f"{split}[{i}] source_assistant mismatch for {exp[0]}")

print("train_processed=%d/%d" % (len(agg["train"]), len(corp["train"])))
print("validation_processed=%d/%d" % (len(agg["validation"]), len(corp["validation"])))
print("total=%d" % (len(agg["train"])+len(agg["validation"])))
print("unique_source_ids=%d" % len(seen))
if errs:
    print("VERIFY=FAIL")
    for e in errs[:50]: print(" -", e)
    sys.exit(1)
print("VERIFY=PASS")
