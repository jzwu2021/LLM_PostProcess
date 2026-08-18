import json
p='/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl'
rows=[json.loads(l) for l in open(p,encoding='utf-8')]
for r in rows[1620:1630]:
    print('=== ID', r.get('id'))
    print('--- USER'); print(r['messages'][0]['content'] if 'messages' in r else r)
    print('--- ASSISTANT'); print((r['messages'][1]['content'] if 'messages' in r else '')[:1500])
