import json
src="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
rows=[json.loads(l) for l in open(src)]
sl=rows[2190:2200]
out=[]
for r in sl:
    m=r["messages"]
    u=[x["content"] for x in m if x["role"]=="user"][0]
    a=[x["content"] for x in m if x["role"]=="assistant"][0]
    out.append({"id":r.get("id"),"user":u,"assistant":a})
print(json.dumps(out,ensure_ascii=False,indent=1)[:20000])
