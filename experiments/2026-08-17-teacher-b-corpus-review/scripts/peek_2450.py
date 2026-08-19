import json
c=[json.loads(l) for l in open('/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl')]
for r in c[2450:2460]:
    m={x['role']:x['content'] for x in r['messages']}
    print('==', r['id'])
    print(m['user'][:700])
    print('--assistant len', len(m['assistant']))
