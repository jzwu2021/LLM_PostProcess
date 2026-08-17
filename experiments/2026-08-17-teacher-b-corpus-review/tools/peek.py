import json, sys
start=int(sys.argv[1]); n=int(sys.argv[2])
lines=open('research/ai-infra-expert/corpus/train.jsonl').read().splitlines()[start:start+n]
for l in lines:
    d=json.loads(l)
    print(json.dumps(d, ensure_ascii=False))
    print('=====')
