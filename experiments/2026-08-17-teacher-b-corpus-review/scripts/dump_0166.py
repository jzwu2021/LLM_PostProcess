import json
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows=[json.loads(l) for i,l in enumerate(open(p,encoding="utf-8")) if 1650<=i<1660]
for r in rows:
    m=r["messages"]
    u=next(x["content"] for x in m if x["role"]=="user")
    a=next(x["content"] for x in m if x["role"]=="assistant")
    print("="*20, r["id"])
    print("USER:", u[:900])
    print("ASST:", a[:700])
