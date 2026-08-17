import json
rows = [json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')][200:210]
for r in rows:
    m = {x['role']: x['content'] for x in r['messages']}
    print('ID', r['id'], '| cat', r.get('category'), '| task', r.get('task_type'), '| diff', r.get('difficulty'), '| concepts', r.get('concepts'))
    print('USER:', m.get('user'))
    print('ASSISTANT:', m.get('assistant'))
    print('=' * 80)
