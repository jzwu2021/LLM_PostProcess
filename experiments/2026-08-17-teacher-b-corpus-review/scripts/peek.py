import json, glob, sys
src=[json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')]
print('corpus_len',len(src))
fs=sorted(glob.glob('experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-*.jsonl'))
done=sum(1 for f in fs for l in open(f) if l.strip())
print('done',done)
last=[json.loads(l) for l in open(fs[-1])][-1]
print('last_done_id',last['source_id'])
print('keys',list(src[0].keys()))
for i in range(done,done+10):
    r=src[i]
    print('=====',i+1,r.get('id'))
    print(json.dumps(r,ensure_ascii=False)[:2600])
