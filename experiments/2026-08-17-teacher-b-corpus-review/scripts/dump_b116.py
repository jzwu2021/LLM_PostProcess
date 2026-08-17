import json, sys
start = int(sys.argv[1]); n = int(sys.argv[2])
rows = [json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')][start:start+n]
for i, r in enumerate(rows):
    m = r['messages']
    u = [x for x in m if x['role'] == 'user'][0]['content']
    a = [x for x in m if x['role'] == 'assistant'][0]['content']
    print('=== IDX', start + i, 'ID', r['id'], '| cat', r.get('category'), '| task', r.get('task_type'), '| diff', r.get('difficulty'))
    print('CONCEPTS:', r.get('concepts'))
    print('USER:', u)
    print('ASSISTANT:', a)
    print()
