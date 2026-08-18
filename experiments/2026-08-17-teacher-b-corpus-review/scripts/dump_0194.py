import json
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
lines=open(p).read().splitlines()[1930:1940]
for l in lines:
    d=json.loads(l)
    m=d["messages"]
    print("ID:", d.get("id"))
    print("U:", m[0]["content"][:900])
    print("A:", m[1]["content"][:500])
    print("=" * 60)
