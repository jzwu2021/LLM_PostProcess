import json, glob, os
R = 'experiments/2026-08-17-teacher-b-corpus-review/results'


def load(pref):
    ids = []
    for f in sorted(glob.glob(os.path.join(R, pref + '-batch-*.jsonl'))):
        for l in open(f):
            l = l.strip()
            if l:
                ids.append(json.loads(l)['source_id'])
    return ids


for name, corpus in (('train', 'research/ai-infra-expert/corpus/train.jsonl'),
                     ('validation', 'research/ai-infra-expert/corpus/validation.jsonl')):
    ids = load(name)
    corp = [json.loads(l) for l in open(corpus) if l.strip()]
    cids = [c['id'] for c in corp]
    print(name, 'processed', len(ids), 'corpus', len(cids), 'prefix_ok', cids[:len(ids)] == ids)
    if name == 'train':
        nxt = corp[len(ids):len(ids) + 10]
        print('NEXT_BATCH_INDEX', len(ids))
        print(json.dumps(nxt, ensure_ascii=False))
