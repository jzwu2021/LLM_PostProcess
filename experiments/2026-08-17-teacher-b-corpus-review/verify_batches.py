import json, os, sys, glob, re

EXP = "experiments/2026-08-17-teacher-b-corpus-review"
RES = os.path.join(EXP, "results")
CORP = {"train": "research/ai-infra-expert/corpus/train.jsonl",
        "validation": "research/ai-infra-expert/corpus/validation.jsonl"}
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
errs = []

corpus = {}
for split, p in CORP.items():
    seq = []
    for l in open(p, encoding="utf-8"):
        d = json.loads(l)
        m = d["messages"]
        u = [x for x in m if x["role"]=="user"][0]["content"]
        a = [x for x in m if x["role"]=="assistant"][0]["content"]
        seq.append((d["id"], u, a))
    corpus[split] = seq

seen = set()
totals = {}
for split in ("train","validation"):
    files = sorted(glob.glob(os.path.join(RES, "%s-batch-*.jsonl" % split)))
    agg = []
    for fp in files:
        raw = open(fp, encoding="utf-8").read()
        lines = [x for x in raw.split("\n") if x.strip()]
        n = 0
        for ln, line in enumerate(lines, 1):
            try:
                r = json.loads(line)
            except Exception as e:
                errs.append("%s:%d JSON parse: %s" % (fp, ln, e)); continue
            n += 1
            for k in REQ:
                if k not in r: errs.append("%s:%d missing field %s" % (fp, ln, k))
            if len(r) != 12: errs.append("%s:%d field count %d" % (fp, ln, len(r)))
            if r.get("teacher_lane") != "teacher-B": errs.append("%s:%d teacher_lane" % (fp, ln))
            if r.get("teacher_model") != "claude-opus-5-current": errs.append("%s:%d teacher_model" % (fp, ln))
            if r.get("calibration_status") != "provisional": errs.append("%s:%d calibration_status" % (fp, ln))
            if r.get("decision") not in ("keep","rewrite","reject"): errs.append("%s:%d decision" % (fp, ln))
            ca = r.get("corrected_answer")
            if not isinstance(ca, str) or not ca.strip(): errs.append("%s:%d corrected_answer empty" % (fp, ln))
            c = r.get("confidence")
            if not isinstance(c,(int,float)) or not (0.0 <= c <= 1.0): errs.append("%s:%d confidence" % (fp, ln))
            qd = r.get("quality_dimensions")
            if not isinstance(qd, dict): errs.append("%s:%d qd not obj" % (fp, ln))
            else:
                for k in ("technical_correctness","instruction_coverage","operational_safety"):
                    v = qd.get(k)
                    if not isinstance(v,int) or isinstance(v,bool) or not (1<=v<=5):
                        errs.append("%s:%d qd.%s" % (fp, ln, k))
            for k in ("risks","evidence_required"):
                v = r.get(k)
                if not isinstance(v, list) or not all(isinstance(x,str) for x in v):
                    errs.append("%s:%d %s not str list" % (fp, ln, k))
            sid = r.get("source_id")
            if sid in seen: errs.append("%s:%d duplicate source_id %s" % (fp, ln, sid))
            seen.add(sid)
            agg.append(r)
        # batch size: all but possibly the last must be 10
        if n != 10 and fp != files[-1]:
            errs.append("%s batch size %d != 10" % (fp, n))
    # prefix check
    src = corpus[split]
    if len(agg) > len(src):
        errs.append("%s: more outputs (%d) than corpus (%d)" % (split, len(agg), len(src)))
    for i, r in enumerate(agg[:len(src)]):
        sid, u, a = src[i]
        if r.get("source_id") != sid:
            errs.append("%s idx %d order mismatch: %s != %s" % (split, i, r.get("source_id"), sid)); break
        if r.get("source_user") != u: errs.append("%s idx %d source_user mismatch" % (split, i))
        if r.get("source_assistant") != a: errs.append("%s idx %d source_assistant mismatch" % (split, i))
    totals[split] = len(agg)

print("train=%d/5399 validation=%d/601 total=%d/6000" % (totals["train"], totals["validation"], totals["train"]+totals["validation"]))
if errs:
    print("SCHEMA_CHECK=FAIL (%d)" % len(errs))
    for e in errs[:40]: print(" -", e)
    sys.exit(1)
print("SCHEMA_CHECK=PASS")
