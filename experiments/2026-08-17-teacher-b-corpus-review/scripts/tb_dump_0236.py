import json
CORPUS="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
corpus=[json.loads(l) for l in open(CORPUS) if l.strip()]
for i,s in enumerate(corpus[2350:2360]):
    m={x["role"]:x["content"] for x in s["messages"]}
    print("="*20,"IDX",2350+i,"ID",s["id"])
    print("[USER]",m["user"][:1400])
    print("[ASSISTANT]",m.get("assistant","")[:900])
