import json
c=[json.loads(l) for l in open("/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl") if l.strip()]
print("CORPUS_TOTAL",len(c))
for r in c[2430:2440]:
    m={x["role"]:x["content"] for x in r["messages"]}
    print("=====",r["id"])
    print("USER:",m["user"][:1000])
    print("ASSIST:",m["assistant"][:600])
