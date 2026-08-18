import json
rows=[json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl',encoding='utf-8')]
for r in rows[1590:1600]:
    m=r['messages']
    u=[x for x in m if x['role']=='user'][0]['content']
    a=[x for x in m if x['role']=='assistant'][0]['content']
    print('ID',r['id'],'|',r.get('category'),'|',r.get('task_type'),'|',r.get('difficulty'))
    print('U:',u[:900])
    print('A_len',len(a))
    print('A:',a[:1200])
    print('='*90)
