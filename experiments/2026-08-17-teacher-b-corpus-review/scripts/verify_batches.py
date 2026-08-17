import json, glob, os, sys, re

ROOT='/home/johnson/workspace/LLM_PostProcess'
EXP=os.path.join(ROOT,'experiments/2026-08-17-teacher-b-corpus-review')
RES=os.path.join(EXP,'results')
FIELDS={"source_id","teacher_lane","teacher_model","calibration_status","decision",
        "source_user","source_assistant","corrected_answer","quality_dimensions",
        "risks","evidence_required","confidence"}
errors=[]

def load_corpus(name):
    recs=[json.loads(l) for l in open(os.path.join(ROOT,'research/ai-infra-expert/corpus',name))]
    out=[]
    for r in recs:
        m=r['messages']
        u=[x for x in m if x['role']=='user'][0]['content']
        a=[x for x in m if x['role']=='assistant'][0]['content']
        out.append((r['id'],u,a))
    return out

corp={'train':load_corpus('train.jsonl'),'validation':load_corpus('validation.jsonl')}

seen={}
for split in ('train','validation'):
    files=sorted(glob.glob(os.path.join(RES,f'{split}-batch-*.jsonl')))
    seq=[]
    for fp in files:
        raw=open(fp,'rb').read().decode('utf-8')
        lines=raw.split('\n')
        if lines and lines[-1]=='': lines.pop()
        if len(lines)==0: errors.append(f'{fp}: empty')
        for i,ln in enumerate(lines,1):
            try: d=json.loads(ln)
            except Exception as e:
                errors.append(f'{fp}:{i} JSON parse: {e}'); continue
            miss=FIELDS-set(d); extra=set(d)-FIELDS
            if miss: errors.append(f'{fp}:{i} missing {sorted(miss)}')
            if extra: errors.append(f'{fp}:{i} extra {sorted(extra)}')
            if d.get('teacher_lane')!='teacher-B': errors.append(f'{fp}:{i} teacher_lane')
            if d.get('teacher_model')!='claude-opus-5-current': errors.append(f'{fp}:{i} teacher_model')
            if d.get('calibration_status')!='provisional': errors.append(f'{fp}:{i} calibration_status')
            if d.get('decision') not in ('keep','rewrite','reject'): errors.append(f'{fp}:{i} decision')
            ca=d.get('corrected_answer')
            if not isinstance(ca,str) or not ca.strip(): errors.append(f'{fp}:{i} corrected_answer empty')
            c=d.get('confidence')
            if not isinstance(c,(int,float)) or isinstance(c,bool) or not (0.0<=c<=1.0):
                errors.append(f'{fp}:{i} confidence')
            qd=d.get('quality_dimensions')
            if not isinstance(qd,dict) or set(qd)!={'technical_correctness','instruction_coverage','operational_safety'}:
                errors.append(f'{fp}:{i} quality_dimensions keys')
            else:
                for k,v in qd.items():
                    if not isinstance(v,int) or isinstance(v,bool) or not (1<=v<=5):
                        errors.append(f'{fp}:{i} quality_dimensions.{k}')
            for k in ('risks','evidence_required'):
                v=d.get(k)
                if not isinstance(v,list) or not all(isinstance(x,str) for x in v):
                    errors.append(f'{fp}:{i} {k} not str list')
            sid=d.get('source_id')
            if sid in seen: errors.append(f'{fp}:{i} duplicate source_id {sid} (also {seen[sid]})')
            else: seen[sid]=f'{fp}:{i}'
            seq.append((sid,d.get('source_user'),d.get('source_assistant')))
    # prefix check
    ref=corp[split]
    if len(seq)>len(ref): errors.append(f'{split}: too many records {len(seq)}>{len(ref)}')
    for i,(sid,u,a) in enumerate(seq):
        if i>=len(ref): break
        rid,ru,ra=ref[i]
        if sid!=rid: errors.append(f'{split}[{i}] id mismatch {sid}!={rid}')
        if u!=ru: errors.append(f'{split}[{i}] source_user mismatch')
        if a!=ra: errors.append(f'{split}[{i}] source_assistant mismatch')
    print(f'{split}: {len(seq)}/{len(ref)} records, {len(files)} batches, prefix OK={not errors}')

if errors:
    print('FAIL', len(errors))
    for e in errors[:40]: print(' ',e)
    sys.exit(1)
print('VERIFICATION PASS: total', len(seen))
