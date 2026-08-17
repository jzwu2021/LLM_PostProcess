import json, sys
start = int(sys.argv[1]); n = int(sys.argv[2])
L = open('research/ai-infra-expert/corpus/train.jsonl').read().splitlines()
print("TOTAL", len(L))
for i in range(start, start+n):
    d = json.loads(L[i])
    print("IDX", i, "KEYS", list(d.keys()))
    print(json.dumps(d, ensure_ascii=False)[:1500])
    print("-----")
