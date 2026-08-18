import json
rows=[json.loads(l) for l in open("/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl") if l.strip()]
print("CORPUS_TOTAL", len(rows))
for r in rows[1840:1850]:
    m={x["role"]:x["content"] for x in r["messages"]}
    print("=====", r["id"])
    print("U:", m["user"][:700].replace("\n"," "))
    print("A:", m["assistant"][:350].replace("\n"," "))
