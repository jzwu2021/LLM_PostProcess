import json
rows=[]
with open("research/ai-infra-expert/corpus/train.jsonl") as f:
    for i,l in enumerate(f):
        if 1540<=i<1550:
            rows.append((i,json.loads(l)))
for i,r in rows:
    print("=== idx",i,"keys",list(r.keys()))
    print(json.dumps(r,ensure_ascii=False)[:1600])
