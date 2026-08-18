import json
ROOT = "/home/johnson/workspace/LLM_PostProcess"
rows = [json.loads(l) for l in open(f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl") if l.strip()]
sel = rows[1970:1980]
for r in sel:
    m = {x["role"]: x["content"] for x in r["messages"]}
    print("==", r["id"], "USERLEN", len(m["user"]), "ASSTLEN", len(m["assistant"]))
    print(m["user"][:400].replace("\n", " "))
print("DISTINCT_USER", len({ {x["role"]:x["content"] for x in r["messages"]}["user"] for r in sel }))
