import json,glob,os,sys
BASE='/home/johnson/workspace/LLM_PostProcess'
# results/ is resolved relative to this script so the verifier can be exercised
# against a sandboxed copy (negative control); corpus stays canonical via BASE.
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results')
err=[]
def corpus(name):
    return [json.loads(l) for l in open(os.path.join(BASE,'research/ai-infra-expert/corpus',name))]
tr=corpus('train.jsonl'); va=corpus('validation.jsonl')
REQ=["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user","source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]
seen=set()
agg={'train':[],'validation':[]}
for fp in sorted(glob.glob(RES+'/*.jsonl')):
    split='train' if os.path.basename(fp).startswith('train') else 'validation'
    lines=open(fp).read().split('\n')
    if lines and lines[-1]=='': lines.pop()
    if len(lines)!=10: err.append(f'{fp}: expected 10 lines got {len(lines)}')
    for i,l in enumerate(lines):
        try: d=json.loads(l)
        except Exception as e: err.append(f'{fp}:{i+1} parse {e}'); continue
        for k in REQ:
            if k not in d: err.append(f'{fp}:{i+1} missing {k}')
        if d.get('teacher_lane')!='teacher-B': err.append(f'{fp}:{i+1} lane')
        if d.get('teacher_model')!='claude-opus-5-current': err.append(f'{fp}:{i+1} model')
        if d.get('calibration_status')!='provisional': err.append(f'{fp}:{i+1} status')
        if d.get('decision') not in ('keep','rewrite','reject'): err.append(f'{fp}:{i+1} decision')
        if not isinstance(d.get('corrected_answer'),str) or not d['corrected_answer'].strip(): err.append(f'{fp}:{i+1} empty corrected_answer')
        c=d.get('confidence')
        if not isinstance(c,(int,float)) or not (0<=c<=1): err.append(f'{fp}:{i+1} confidence')
        qd=d.get('quality_dimensions',{})
        for k in ('technical_correctness','instruction_coverage','operational_safety'):
            v=qd.get(k)
            if not isinstance(v,int) or not (1<=v<=5): err.append(f'{fp}:{i+1} qd {k}')
        if not isinstance(d.get('risks'),list) or not isinstance(d.get('evidence_required'),list): err.append(f'{fp}:{i+1} list fields')
        sid=d.get('source_id')
        if sid in seen: err.append(f'{fp}:{i+1} dup id {sid}')
        seen.add(sid)
        agg[split].append(d)
for split,src in (('train',tr),('validation',va)):
    a=agg[split]
    if len(a)>len(src): err.append(f'{split}: more rows than corpus'); continue
    for i,d in enumerate(a):
        s=src[i]
        if d['source_id']!=s['id']: err.append(f'{split}[{i}] id mismatch {d["source_id"]} vs {s["id"]}'); break
        if d['source_user']!=s['messages'][1]['content']: err.append(f'{split}[{i}] user mismatch')
        if d['source_assistant']!=s['messages'][2]['content']: err.append(f'{split}[{i}] assistant mismatch')
print('train',len(agg['train']),'validation',len(agg['validation']),'total',len(agg['train'])+len(agg['validation']))
if err:
    print('FAIL'); [print(' ',e) for e in err[:40]]; sys.exit(1)
print('PASS')
