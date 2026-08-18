import json
P="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
with open(P,encoding="utf-8") as f:
    for i,l in enumerate(f):
        if 1710<=i<1720:
            r=json.loads(l)
            u=next(x["content"] for x in r["messages"] if x["role"]=="user")
            a=next(x["content"] for x in r["messages"] if x["role"]=="assistant")
            print("###",i,r["id"],r.get("category"))
            print("USER:",u[-900:])
            print("ASST_LEN:",len(a),"ASST_HEAD:",a[:300].replace("\n"," | "))
            print()
