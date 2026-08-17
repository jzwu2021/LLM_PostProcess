import json, glob, sys
BASE='/home/johnson/workspace/LLM_PostProcess'
RES=BASE+'/experiments/2026-08-17-teacher-b-corpus-review/results'
def load(split):
    return [json.loads(l) for l in open(f'{BASE}/research/ai-infra-expert/corpus/{split}.jsonl')]
def done(split):
    ids=[]
    for f in sorted(glob.glob(f'{RES}/{split}-batch-*.jsonl')):
        for l in open(f):
            if l.strip(): ids.append(json.loads(l)['source_id'])
    return ids
tr=load('train'); va=load('validation')
dtr=done('train'); dva=done('validation')
print('TRAIN_DONE',len(dtr),'of',len(tr))
print('VAL_DONE',len(dva),'of',len(va))
print('PREFIX_TRAIN',[r['id'] for r in tr[:len(dtr)]]==dtr)
print('PREFIX_VAL',[r['id'] for r in va[:len(dva)]]==dva)
if len(dtr)<len(tr):
    split='train'; nxt=tr[len(dtr):len(dtr)+10]
else:
    split='validation'; nxt=va[len(dva):len(dva)+10]
n=len(glob.glob(f'{RES}/{split}-batch-*.jsonl'))+1
print('NEXT_SPLIT',split)
print('NEXT_FILE',f'{split}-batch-{n:04d}.jsonl')
print('---')
for r in nxt:
    print(json.dumps(r,ensure_ascii=False))
