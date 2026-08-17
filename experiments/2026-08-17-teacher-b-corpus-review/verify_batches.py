import json,glob,os,re,sys
ROOT="/home/johnson/workspace/LLM_PostProcess"
EXP=os.path.join(ROOT,"experiments/2026-08-17-teacher-b-corpus-review")
REQ=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user","source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]
errs=[]
def load_corpus(p):
    out=[]
    for l in open(p):
        d=json.loads(l); m={x["role"]:x["content"] for x in d["messages"]}
        out.append((d["id"],m["user"],m["assistant"]))
    return out
corp={"train":load_corpus(os.path.join(ROOT,"research/ai-infra-expert/corpus/train.jsonl")),
      "validation":load_corpus(os.path.join(ROOT,"research/ai-infra-expert/corpus/validation.jsonl"))}
seen={}
seq={"train":[],"validation":[]}
for split in ("train","validation"):
    files=sorted(glob.glob(os.path.join(EXP,"results",f"{split}-batch-*.jsonl")))
    for fp in files:
        n=0
        for i,line in enumerate(open(fp),1):
            if not line.strip(): errs.append(f"{fp}:{i} blank"); continue
            try: r=json.loads(line)
            except Exception as e: errs.append(f"{fp}:{i} bad json {e}"); continue
            n+=1
            for k in REQ:
                if k not in r: errs.append(f"{fp}:{i} missing {k}")
            if r.get("teacher_lane")!="teacher-B": errs.append(f"{fp}:{i} lane")
            if r.get("teacher_model")!="claude-opus-5-current": errs.append(f"{fp}:{i} model")
            if r.get("calibration_status")!="provisional": errs.append(f"{fp}:{i} status")
            if r.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{fp}:{i} decision")
            if not isinstance(r.get("corrected_answer"),str) or not r["corrected_answer"].strip(): errs.append(f"{fp}:{i} empty corrected")
            c=r.get("confidence")
            if not isinstance(c,(int,float)) or not (0<=c<=1): errs.append(f"{fp}:{i} confidence")
            qd=r.get("quality_dimensions",{})
            for k in ("technical_correctness","instruction_coverage","operational_safety"):
                v=qd.get(k)
                if not isinstance(v,int) or not (1<=v<=5): errs.append(f"{fp}:{i} qd {k}")
            for k in ("risks","evidence_required"):
                if not isinstance(r.get(k),list) or not all(isinstance(x,str) for x in r.get(k,[])): errs.append(f"{fp}:{i} {k}")
            sid=r.get("source_id")
            if sid in seen: errs.append(f"{fp}:{i} dup {sid} (also {seen[sid]})")
            seen[sid]=f"{fp}:{i}"
            seq[split].append((sid,r.get("source_user"),r.get("source_assistant")))
        if n!=10 and fp!=files[-1]: errs.append(f"{fp} count {n}")
    # prefix check
    ref=corp[split]
    if len(seq[split])>len(ref): errs.append(f"{split} longer than corpus")
    for j,(sid,u,a) in enumerate(seq[split]):
        if j>=len(ref): break
        if (sid,u,a)!=ref[j]: errs.append(f"{split} idx {j} mismatch {sid} vs {ref[j][0]}")
print("train",len(seq["train"]),"validation",len(seq["validation"]),"total",len(seq["train"])+len(seq["validation"]))
if errs:
    print("FAIL",len(errs)); [print(" ",e) for e in errs[:30]]; sys.exit(1)
print("PASS")
