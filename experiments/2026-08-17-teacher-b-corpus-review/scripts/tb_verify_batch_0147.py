import json,glob,os,sys
ROOT='/home/johnson/workspace/LLM_PostProcess'
EXP=f'{ROOT}/experiments/2026-08-17-teacher-b-corpus-review'
err=[]
def corpus(name):
    return [json.loads(l) for l in open(f'{ROOT}/research/ai-infra-expert/corpus/{name}.jsonl')]
tr=corpus('train'); va=corpus('validation')
def um(r):
    return ([m for m in r['messages'] if m['role']=='user'][0]['content'],
            [m for m in r['messages'] if m['role']=='assistant'][0]['content'])
REQ=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user","source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]
seen=set()
for kind,src in (("train",tr),("validation",va)):
    files=sorted(glob.glob(f'{EXP}/results/{kind}-batch-*.jsonl'))
    seq=[]
    for fp in files:
        lines=open(fp).read().split("\n")
        lines=[l for l in lines if l.strip()]
        if len(lines)!=10 and fp!=files[-1]: err.append(f"{fp}: {len(lines)} lines")
        for l in lines:
            r=json.loads(l)
            for k in REQ:
                if k not in r: err.append(f"{fp}:{r.get('source_id')} missing {k}")
            if r["teacher_lane"]!="teacher-B": err.append(f"lane {fp}")
            if r["teacher_model"]!="claude-opus-5-current": err.append(f"model {fp}")
            if r["calibration_status"]!="provisional": err.append(f"status {fp}")
            if r["decision"] not in ("keep","rewrite","reject"): err.append(f"decision {fp}")
            if not isinstance(r["corrected_answer"],str) or not r["corrected_answer"].strip(): err.append(f"empty ca {r['source_id']}")
            if not (0<=float(r["confidence"])<=1): err.append(f"conf {r['source_id']}")
            qd=r["quality_dimensions"]
            for d in ("technical_correctness","instruction_coverage","operational_safety"):
                if not isinstance(qd.get(d),int) or not 1<=qd[d]<=5: err.append(f"qd {d} {r['source_id']}")
            if not isinstance(r["risks"],list) or not isinstance(r["evidence_required"],list): err.append(f"list {r['source_id']}")
            if r["source_id"] in seen: err.append(f"dup {r['source_id']}")
            seen.add(r["source_id"])
            seq.append(r)
    for i,r in enumerate(seq):
        c=src[i]
        u,a=um(c)
        if r["source_id"]!=c["id"]: err.append(f"{kind} order idx{i}: {r['source_id']} vs {c['id']}")
        if r["source_user"]!=u: err.append(f"{kind} user mismatch {c['id']}")
        if r["source_assistant"]!=a: err.append(f"{kind} assistant mismatch {c['id']}")
    print(kind,"count",len(seq))
print("ERRORS",len(err))
for e in err[:20]: print(e)
sys.exit(1 if err else 0)
