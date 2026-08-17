import json, sys
start = int(sys.argv[1]); n = int(sys.argv[2])
rows = [json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')][start:start+n]
for r in rows:
    u = [m['content'] for m in r['messages'] if m['role'] == 'user'][0]
    a = [m['content'] for m in r['messages'] if m['role'] == 'assistant'][0]
    print(json.dumps({'id': r['id'], 'cat': r.get('category'), 'concepts': r.get('concepts'), 'user': u, 'assistant': a}, ensure_ascii=False))
