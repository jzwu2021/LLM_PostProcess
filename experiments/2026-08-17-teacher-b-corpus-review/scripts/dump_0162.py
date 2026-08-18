import json
p='/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl'
rows=[json.loads(l) for l in open(p)]
print('TOTAL',len(rows))
for r in rows[1610:1620]:
    print('=====ID',r.get('id'))
    print('KEYS',list(r.keys()))
    m=r.get('messages')
    if m:
        for x in m:
            print('--',x['role'],'--')
            print(x['content'])
    else:
        print(json.dumps(r,ensure_ascii=False)[:3000])
