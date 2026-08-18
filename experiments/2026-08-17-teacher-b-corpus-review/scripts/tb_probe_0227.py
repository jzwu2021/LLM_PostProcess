import json
CORPUS = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows = [json.loads(l) for l in open(CORPUS) if l.strip()]
print("TOTAL", len(rows))
for r in rows[2260:2270]:
    m = {x["role"]: x["content"] for x in r["messages"]}
    print("---", r["id"])
    print("U:", m["user"])
    print("A:", m["assistant"][:300])
