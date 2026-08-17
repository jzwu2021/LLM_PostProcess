import json,glob,sys
base='experiments/2026-08-17-teacher-b-corpus-review/results/'
ids=[]
for f in sorted(glob.glob(base+'train-batch-*.jsonl')):
    for l in open(f):
        if l.strip(): ids.append(json.loads(l)['source_id'])
n=len(ids)
corpus=[json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl') if l.strip()]
print('processed_train',n,'corpus',len(corpus))
print(json.dumps(corpus[n:n+10],ensure_ascii=False,indent=1))
