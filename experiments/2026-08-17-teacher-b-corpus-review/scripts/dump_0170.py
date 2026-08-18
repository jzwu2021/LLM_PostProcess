import json
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows=[json.loads(l) for l in open(p,encoding="utf-8")]
for r in rows[1690:1700]:
    u=next(m["content"] for m in r["messages"] if m["role"]=="user")
    print("=== ",r["id"],"|",r["category"],"|",r.get("task_type"))
    print(u)
    print()
