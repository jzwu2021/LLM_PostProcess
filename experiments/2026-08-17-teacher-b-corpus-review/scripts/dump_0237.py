import json
CORPUS = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
print("TOTAL", len(corpus))
for i, s in enumerate(corpus[2360:2370], start=2360):
    m = {x["role"]: x["content"] for x in s["messages"]}
    print("=" * 70)
    print("IDX", i, "ID", s["id"])
    print("--USER--")
    print(m["user"][:1800])
    print("--ASSISTANT (len %d)--" % len(m.get("assistant","")))
    print(m.get("assistant","")[:1200])
