import json
CORPUS = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows = [json.loads(l) for l in open(CORPUS) if l.strip()]
print("TOTAL", len(rows))
for i in range(2290, 2300):
    r = rows[i]
    m = {x["role"]: x["content"] for x in r["messages"]}
    print("=== idx", i, "id", r["id"])
    print("USER:", m["user"][:700])
    print("ASSISTANT:", m["assistant"][:500])
