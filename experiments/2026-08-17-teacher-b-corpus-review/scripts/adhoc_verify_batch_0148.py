import json,glob,os,sys
BASE="/home/johnson/workspace/LLM_PostProcess"
EXP=os.path.join(BASE,"experiments/2026-08-17-teacher-b-corpus-review")
CORP=os.path.join(BASE,"research/ai-infra-expert/corpus")
errs=[]

def load_corpus(name):
    rows=[]
    with open(os.path.join(CORP,name)) as f:
        for line in f:
            d=json.loads(line)
            u=a=""
            for m in d.get("messages",[]):
                if m["role"]=="user" and not u: u=m["content"]
                elif m["role"]=="assistant" and not a: a=m["content"]
            rows.append((d.get("id"),u,a))
    return rows

train=load_corpus("train.jsonl"); val=load_corpus("validation.jsonl")
REQ=["source_id","teacher_lane","teacher_model","calibration_status","decision",
     "source_user","source_assistant","corrected_answer","quality_dimensions",
     "risks","evidence_required","confidence"]

def read_split(prefix):
    files=sorted(glob.glob(os.path.join(EXP,"results",prefix+"-batch-*.jsonl")))
    recs=[]
    for fp in files:
        raw=open(fp).read()
        lines=[l for l in raw.split("\n") if l.strip()]
        if len(lines)!=10 and fp==files[-1]:
            pass
        for ln,l in enumerate(lines,1):
            try: d=json.loads(l)
            except Exception as e:
                errs.append(f"{fp}:{ln} JSON parse: {e}"); continue
            recs.append((fp,ln,d))
    return files,recs

allids=set()
for prefix,corpus in (("train",train),("validation",val)):
    files,recs=read_split(prefix)
    for fp,ln,d in recs:
        for k in REQ:
            if k not in d: errs.append(f"{fp}:{ln} missing {k}")
        if d.get("teacher_lane")!="teacher-B": errs.append(f"{fp}:{ln} bad lane")
        if d.get("teacher_model")!="claude-opus-5-current": errs.append(f"{fp}:{ln} bad model")
        if d.get("calibration_status")!="provisional": errs.append(f"{fp}:{ln} bad status")
        if d.get("decision") not in ("keep","rewrite","reject"): errs.append(f"{fp}:{ln} bad decision")
        if not isinstance(d.get("corrected_answer"),str) or not d["corrected_answer"].strip():
            errs.append(f"{fp}:{ln} empty corrected_answer")
        c=d.get("confidence")
        if not isinstance(c,(int,float)) or not (0.0<=c<=1.0): errs.append(f"{fp}:{ln} bad confidence")
        qdv=d.get("quality_dimensions")
        if not isinstance(qdv,dict): errs.append(f"{fp}:{ln} qd not object")
        else:
            for k in ("technical_correctness","instruction_coverage","operational_safety"):
                v=qdv.get(k)
                if not isinstance(v,int) or not (1<=v<=5): errs.append(f"{fp}:{ln} bad qd {k}")
        for k in ("risks","evidence_required"):
            v=d.get(k)
            if not isinstance(v,list) or not all(isinstance(x,str) for x in v): errs.append(f"{fp}:{ln} bad {k}")
        sid=d.get("source_id")
        if sid in allids: errs.append(f"{fp}:{ln} duplicate source_id {sid}")
        allids.add(sid)
    # prefix check
    if len(recs)>len(corpus): errs.append(f"{prefix}: more records than corpus")
    for i,(fp,ln,d) in enumerate(recs):
        cid,cu,ca=corpus[i]
        if d.get("source_id")!=cid: errs.append(f"{prefix}[{i}] id mismatch {d.get('source_id')} != {cid}")
        if d.get("source_user")!=cu: errs.append(f"{prefix}[{i}] source_user mismatch")
        if d.get("source_assistant")!=ca: errs.append(f"{prefix}[{i}] source_assistant mismatch")
    print(f"{prefix}: {len(recs)} records over {len(files)} files")

print("ERRORS:",len(errs))
for e in errs[:20]: print(" ",e)
sys.exit(1 if errs else 0)
