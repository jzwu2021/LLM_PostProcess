import json
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
lines=open(p,encoding="utf-8").readlines()[1390:1400]
for l in lines:
    d=json.loads(l)
    u=[m["content"] for m in d["messages"] if m["role"]=="user"][0]
    a=[m["content"] for m in d["messages"] if m["role"]=="assistant"][0]
    print("=== ", d["id"], " ulen",len(u)," alen",len(a))
    print(u[:700])
    print("--- assistant head:", a[:300].replace("\n"," "))
