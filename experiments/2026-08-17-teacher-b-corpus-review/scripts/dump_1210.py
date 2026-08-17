import json
rows = [json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')]
for i, r in enumerate(rows[1210:1220], 1210):
    m = r['messages']
    u = [x['content'] for x in m if x['role'] == 'user'][0]
    a = [x['content'] for x in m if x['role'] == 'assistant'][0]
    print('===', i, r['id'], '|', r.get('category'), '|', r.get('task_type'), '|', r.get('difficulty'))
    print('CONCEPTS:', r.get('concepts'))
    print('USER:', u)
    print('ASST:', a)
    print()
