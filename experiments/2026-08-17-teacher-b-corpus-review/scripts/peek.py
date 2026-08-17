import json, sys
lo, hi = int(sys.argv[1]), int(sys.argv[2])
lines = open('research/ai-infra-expert/corpus/train.jsonl').read().splitlines()
for i in range(lo, hi):
    d = json.loads(lines[i])
    print('=== IDX', i, 'KEYS', list(d.keys()))
    print(json.dumps(d, ensure_ascii=False)[:2600])
    print()
