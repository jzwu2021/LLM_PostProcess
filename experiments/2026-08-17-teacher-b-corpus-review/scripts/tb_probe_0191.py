import json
ROOT = "/home/johnson/workspace/LLM_PostProcess"
rows = [json.loads(l) for l in open(f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl") if l.strip()]
print("TOTAL", len(rows))
for r in rows[1900:1910]:
    m = {x["role"]: x["content"] for x in r["messages"]}
    print("==", r["id"])
    print("U:", m["user"][:600].replace("\n", " "))
    print("A_LEN:", len(m["assistant"]), "A_HEAD:", m["assistant"][:300].replace("\n", " "))
