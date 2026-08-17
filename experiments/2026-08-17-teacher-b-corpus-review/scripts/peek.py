import json, sys
off = int(sys.argv[1]); n = int(sys.argv[2])
rows = [json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')]
print('TOTAL', len(rows))
for r in rows[off:off+n]:
    print('===ID', r.get('id'))
    print('KEYS', list(r.keys()))
    print('USER:', str(r.get('user', ''))[:700])
    print('ASST:', str(r.get('assistant', ''))[:900])
