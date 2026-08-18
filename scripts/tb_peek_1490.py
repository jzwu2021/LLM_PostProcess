import json
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows=[]
with open(p) as f:
    for i,l in enumerate(f):
        if 1480<=i<1490: rows.append(json.loads(l))
        elif i>=1490: break
print("KEYS", list(rows[0].keys()))
for r in rows:
    print("="*60)
    print(json.dumps(r, ensure_ascii=False)[:2600])
