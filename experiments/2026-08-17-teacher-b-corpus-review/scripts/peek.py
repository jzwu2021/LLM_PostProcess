import json, sys
start = int(sys.argv[1]); n = int(sys.argv[2])
rows = [json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')][start:start+n]
for r in rows:
    print('=== KEYS:', list(r.keys()))
    print('ID:', r.get('id'))
    print('USER:', r.get('user', '')[:1500])
    print('ASSISTANT:', r.get('assistant', '')[:2200])
    print()
