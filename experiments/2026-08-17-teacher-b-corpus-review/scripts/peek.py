import json, sys
lines = open('research/ai-infra-expert/corpus/train.jsonl').read().splitlines()
print('train_total', len(lines))
print('keys', list(json.loads(lines[0]).keys()))
prev = open('experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0122.jsonl').read().splitlines()[-1]
print('prev_last_id', json.loads(prev)['source_id'])
start = 1220
for i in range(start, start + 10):
    d = json.loads(lines[i])
    print('---INDEX', i, 'ID', d['id'])
    print(json.dumps(d, ensure_ascii=False)[:4000])
