import json
ROOT = "/home/johnson/workspace/LLM_PostProcess"
rows = [json.loads(l) for l in open(f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl") if l.strip()]
print("TOTAL", len(rows))
for r in rows[2250:2260]:
    m = {x["role"]: x["content"] for x in r["messages"]}
    print("=== id", r["id"])
    print("USER:", m["user"][:400].replace("\n", " | "))
    print("ASST:", m["assistant"][:300].replace("\n", " | "))
