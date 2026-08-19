import json
ROOT="/home/johnson/workspace/LLM_PostProcess"
corpus=[json.loads(l) for l in open(f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl") if l.strip()]
print("TOTAL", len(corpus))
for i,s in enumerate(corpus[2380:2390], start=2380):
    m={x["role"]:x["content"] for x in s["messages"]}
    print("="*70)
    print("IDX",i,"ID",s["id"])
    print("--USER--")
    print(m["user"][:1800])
    print("--ASSISTANT (len %d)--"%len(m["assistant"]))
    print(m["assistant"][:900])
