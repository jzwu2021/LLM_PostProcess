import json
CORPUS="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
START,COUNT=1700,10
with open(CORPUS,encoding="utf-8") as f:
    rows=[json.loads(l) for i,l in enumerate(f) if START<=i<START+COUNT]
print(len(rows))
for r in rows:
    m=r["messages"]
    su=next(x["content"] for x in m if x["role"]=="user")
    sa=next(x["content"] for x in m if x["role"]=="assistant")
    print("="*70)
    print("ID",r["id"])
    print("USER:",su[:600])
    print("ASST[%d]:"%len(sa),sa[:400])
