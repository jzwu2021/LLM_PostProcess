import json
rows=[json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')][1630:1640]
for d in rows:
    m=d['messages']
    u=[x for x in m if x['role']=='user'][0]['content']
    a=[x for x in m if x['role']=='assistant'][0]['content']
    print('=====',d['id'],'|',d.get('category'),'|',d.get('task_type'),'|',d.get('difficulty'))
    print('--USER--'); print(u)
    print('--ASSISTANT--'); print(a)
