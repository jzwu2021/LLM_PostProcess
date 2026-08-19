import json
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows=[json.loads(l) for l in open(p) if l.strip()]
for i,r in enumerate(rows[2340:2342]):
    m={x["role"]:x["content"] for x in r["messages"]}
    print("=== IDX",2340+i,"ID",r.get("id"))
    print("USER:",m["user"])
    print("ASSISTANT:",m["assistant"])
    print()
