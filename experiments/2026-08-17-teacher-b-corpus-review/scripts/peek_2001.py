import json
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows=[json.loads(l) for l in open(p)]
for r in rows[2000:2010]:
    m=r["messages"]
    u=[x["content"] for x in m if x["role"]=="user"][0]
    a=[x["content"] for x in m if x["role"]=="assistant"][0]
    print("=== ID",r.get("id"))
    print("U:",u)
    print("A_LEN:",len(a))
