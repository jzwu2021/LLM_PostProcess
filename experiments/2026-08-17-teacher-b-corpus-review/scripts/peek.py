import json, sys
lines = open('research/ai-infra-expert/corpus/train.jsonl').read().splitlines()
print('corpus_total', len(lines))
start = int(sys.argv[1]); n = int(sys.argv[2])
for i in range(start, start+n):
    d = json.loads(lines[i])
    print('=== idx', i, 'keys', list(d.keys()))
    print(json.dumps(d, ensure_ascii=False)[:1600])
