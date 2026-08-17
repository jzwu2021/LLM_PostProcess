import json, sys
start = int(sys.argv[1]); n = int(sys.argv[2])
rows = [json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')]
print('TOTAL', len(rows))
print('KEYS', list(rows[0].keys()))
for r in rows[start:start+n]:
    print(json.dumps(r, ensure_ascii=False))
