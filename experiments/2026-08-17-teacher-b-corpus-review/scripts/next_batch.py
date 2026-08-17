import json, glob, sys

RES = 'experiments/2026-08-17-teacher-b-corpus-review/results'

def load(prefix):
    ids = []
    for f in sorted(glob.glob(f'{RES}/{prefix}-batch-*.jsonl')):
        for l in open(f):
            l = l.strip()
            if l:
                ids.append(json.loads(l)['source_id'])
    return ids

tr = load('train')
va = load('validation')
src_tr = [json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')]
src_va = [json.loads(l) for l in open('research/ai-infra-expert/corpus/validation.jsonl')]

print('train_done', len(tr), 'validation_done', len(va))
print('train_prefix_ok', [s['id'] for s in src_tr[:len(tr)]] == tr)
print('validation_prefix_ok', [s['id'] for s in src_va[:len(va)]] == va)

if len(tr) < len(src_tr):
    lane, nxt, n = 'train', src_tr[len(tr):len(tr)+10], len(glob.glob(f'{RES}/train-batch-*.jsonl'))+1
else:
    lane, nxt, n = 'validation', src_va[len(va):len(va)+10], len(glob.glob(f'{RES}/validation-batch-*.jsonl'))+1
print('NEXT_LANE', lane, 'BATCHNUM', '%04d' % n)
with open('/tmp/tb_next.json', 'w') as f:
    json.dump({'lane': lane, 'batch': '%04d' % n, 'items': nxt}, f, ensure_ascii=False)
for it in nxt:
    print('---ID', it['id'])
