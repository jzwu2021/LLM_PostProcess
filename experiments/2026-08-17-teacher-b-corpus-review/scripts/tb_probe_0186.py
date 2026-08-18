import json
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows=[json.loads(l) for l in open(p) if l.strip()]
print("TOTAL", len(rows))
for r in rows[1850:1860]:
    m={x["role"]:x["content"] for x in r["messages"]}
    print("==", r["id"])
    print("U:", m["user"][:700].replace("\n"," "))
    print("A:", m["assistant"][:350].replace("\n"," "))
