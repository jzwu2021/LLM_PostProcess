import json, glob, sys
BASE='/home/johnson/workspace/LLM_PostProcess/'
RES=BASE+'experiments/2026-08-17-teacher-b-corpus-review/results/'
tr=[json.loads(l) for l in open(BASE+'research/ai-infra-expert/corpus/train.jsonl')]
va=[json.loads(l) for l in open(BASE+'research/ai-infra-expert/corpus/validation.jsonl')]
def done(pref):
    out=[]
    for f in sorted(glob.glob(RES+pref+'-batch-*.jsonl')):
        for l in open(f):
            out.append(json.loads(l)['source_id'])
    return out
dt=done('train'); dv=done('validation')
print('TRAIN_DONE',len(dt),'VAL_DONE',len(dv),'TOTAL',len(dt)+len(dv))
print('train_prefix_ok',[r['id'] for r in tr][:len(dt)]==dt)
print('val_prefix_ok',[r['id'] for r in va][:len(dv)]==dv)
if len(dt)<len(tr):
    src=tr; off=len(dt); lane='train'
else:
    src=va; off=len(dv); lane='validation'
nxt=src[off:off+10]
print('LANE',lane,'NEXTBATCH',len(glob.glob(RES+lane+'-batch-*.jsonl'))+1)
for r in nxt:
    m=r['messages']
    u=[x['content'] for x in m if x['role']=='user'][0]
    a=[x['content'] for x in m if x['role']=='assistant'][0]
    print('=====',r['id'],r['category'],r.get('difficulty'),r.get('concepts'))
    print('U:',u)
    print('A:',a)
