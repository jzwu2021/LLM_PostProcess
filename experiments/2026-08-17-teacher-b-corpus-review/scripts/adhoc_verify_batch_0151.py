import json, glob, os, sys, re
BASE='experiments/2026-08-17-teacher-b-corpus-review/results'
CORPUS={'train':'research/ai-infra-expert/corpus/train.jsonl','validation':'research/ai-infra-expert/corpus/validation.jsonl'}
FIELDS=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user","source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]
errs=[]
def ua(r):
    u=a=''
    for m in r['messages']:
        if m['role']=='user':u=m['content']
        elif m['role']=='assistant':a=m['content']
    return u,a
corp={}
for k,p in CORPUS.items():
    corp[k]=[json.loads(l) for l in open(p)]

seen={}
counts={}
for split in ['train','validation']:
    files=sorted(glob.glob(f'{BASE}/{split}-batch-*.jsonl'))
    seq=[]
    for fp in files:
        raw=open(fp,'rb').read().decode()
        lines=[l for l in raw.split('\n') if l.strip()]
        if len(lines)!=10 and fp==files[-1]:
            pass
        for ln,line in enumerate(lines,1):
            try: rec=json.loads(line)
            except Exception as e: errs.append(f'{fp}:{ln} parse {e}'); continue
            for f in FIELDS:
                if f not in rec: errs.append(f'{fp}:{ln} missing {f}')
            if rec.get('teacher_lane')!='teacher-B': errs.append(f'{fp}:{ln} lane')
            if rec.get('teacher_model')!='claude-opus-5-current': errs.append(f'{fp}:{ln} model')
            if rec.get('calibration_status')!='provisional': errs.append(f'{fp}:{ln} status')
            if rec.get('decision') not in ('keep','rewrite','reject'): errs.append(f'{fp}:{ln} decision')
            if not isinstance(rec.get('corrected_answer'),str) or not rec['corrected_answer'].strip(): errs.append(f'{fp}:{ln} empty corrected')
            c=rec.get('confidence')
            if not isinstance(c,(int,float)) or not (0<=c<=1): errs.append(f'{fp}:{ln} confidence')
            qd=rec.get('quality_dimensions',{})
            for d in ['technical_correctness','instruction_coverage','operational_safety']:
                v=qd.get(d)
                if not isinstance(v,int) or not (1<=v<=5): errs.append(f'{fp}:{ln} qd {d}')
            if not isinstance(rec.get('risks'),list) or not all(isinstance(x,str) for x in rec['risks']): errs.append(f'{fp}:{ln} risks')
            if not isinstance(rec.get('evidence_required'),list) or not all(isinstance(x,str) for x in rec['evidence_required']): errs.append(f'{fp}:{ln} evidence')
            sid=rec.get('source_id')
            if sid in seen: errs.append(f'{fp}:{ln} duplicate source_id {sid} (also {seen[sid]})')
            seen[sid]=f'{fp}:{ln}'
            seq.append(rec)
    counts[split]=len(seq)
    # prefix check
    for i,rec in enumerate(seq):
        src=corp[split][i]
        if rec['source_id']!=src['id']: errs.append(f'{split} idx{i} id mismatch {rec["source_id"]} vs {src["id"]}'); continue
        u,a=ua(src)
        if rec['source_user']!=u: errs.append(f'{split} idx{i} source_user mismatch')
        if rec['source_assistant']!=a: errs.append(f'{split} idx{i} source_assistant mismatch')
print('train',counts['train'],'validation',counts['validation'],'total',sum(counts.values()))
print('errors',len(errs))
for e in errs[:20]: print(e)
sys.exit(1 if errs else 0)
