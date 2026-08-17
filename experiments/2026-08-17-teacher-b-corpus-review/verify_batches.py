import json, os, sys, glob

ROOT = "/home/johnson/workspace/LLM_PostProcess"
RES = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results")
REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision",
       "source_user","source_assistant","corrected_answer","quality_dimensions",
       "risks","evidence_required","confidence"]
errs = []

def load_corpus(p):
    out=[]
    for l in open(p):
        d=json.loads(l)
        m={x["role"]:x["content"] for x in d["messages"]}
        out.append((d["id"], m["user"], m["assistant"]))
    return out

train = load_corpus(os.path.join(ROOT,"research/ai-infra-expert/corpus/train.jsonl"))
val   = load_corpus(os.path.join(ROOT,"research/ai-infra-expert/corpus/validation.jsonl"))

def read_lane(prefix):
    recs=[]
    for f in sorted(glob.glob(os.path.join(RES, prefix+"-batch-*.jsonl"))):
        n=0
        for i,l in enumerate(open(f),1):
            l=l.rstrip("\n")
            if not l.strip(): errs.append(f"{f}:{i} blank line"); continue
            try: d=json.loads(l)
            except Exception as e: errs.append(f"{f}:{i} JSON parse fail {e}"); continue
            recs.append((f,i,d)); n+=1
        if n!=10: errs.append(f"{f} has {n} records, expected 10")
    return recs

BATCH = os.environ.get("BATCH_FILE","")
all_ids=set()
for prefix, corpus in (("train",train),("validation",val)):
    recs = read_lane(prefix)
    if len(recs) > len(corpus):
        errs.append(f"{prefix}: {len(recs)} records exceed corpus {len(corpus)}")
    for idx,(f,i,d) in enumerate(recs):
        tag=f"{os.path.basename(f)}:{i}"
        miss=[k for k in REQ if k not in d]
        extra=[k for k in d if k not in REQ]
        if miss: errs.append(f"{tag} missing fields {miss}")
        if extra: errs.append(f"{tag} unexpected fields {extra}")
        if d.get("teacher_lane")!="teacher-B": errs.append(f"{tag} bad teacher_lane")
        if d.get("teacher_model")!="claude-opus-5-current": errs.append(f"{tag} bad teacher_model")
        if d.get("calibration_status")!="provisional": errs.append(f"{tag} bad calibration_status")
        if d.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{tag} bad decision")
        ca=d.get("corrected_answer")
        if not isinstance(ca,str) or not ca.strip(): errs.append(f"{tag} empty corrected_answer")
        c=d.get("confidence")
        if not isinstance(c,(int,float)) or not (0.0<=c<=1.0): errs.append(f"{tag} confidence out of range")
        qd=d.get("quality_dimensions")
        if not isinstance(qd,dict): errs.append(f"{tag} quality_dimensions not object")
        else:
            for k in ("technical_correctness","instruction_coverage","operational_safety"):
                v=qd.get(k)
                if not isinstance(v,int) or isinstance(v,bool) or not (1<=v<=5):
                    errs.append(f"{tag} quality_dimensions.{k} invalid: {v!r}")
            if set(qd)!={"technical_correctness","instruction_coverage","operational_safety"}:
                errs.append(f"{tag} quality_dimensions keys wrong")
        for k in ("risks","evidence_required"):
            v=d.get(k)
            if not isinstance(v,list) or not all(isinstance(x,str) for x in v):
                errs.append(f"{tag} {k} not string array")
        sid=d.get("source_id")
        if sid in all_ids: errs.append(f"{tag} duplicate source_id {sid}")
        all_ids.add(sid)
        cid,cu,ca_src = corpus[idx]
        if sid!=cid: errs.append(f"{tag} prefix-order break: got {sid} expected {cid} at corpus index {idx}")
        if d.get("source_user")!=cu: errs.append(f"{tag} source_user mismatch")
        if d.get("source_assistant")!=ca_src: errs.append(f"{tag} source_assistant mismatch")
    print(f"{prefix}: {len(recs)} records checked against corpus of {len(corpus)}")

print("unique source_ids:", len(all_ids))
if errs:
    print("VERIFY=FAIL")
    for e in errs[:50]: print(" -", e)
    sys.exit(1)
print("VERIFY=PASS")
