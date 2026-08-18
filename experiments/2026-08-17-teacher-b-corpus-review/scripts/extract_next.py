import json, sys
CORPUS="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
start=int(sys.argv[1]); n=int(sys.argv[2])
rows=[json.loads(l) for l in open(CORPUS)][start:start+n]
out=[]
for r in rows:
    msgs=r.get("messages")
    u=next(m["content"] for m in msgs if m["role"]=="user")
    a=next(m["content"] for m in msgs if m["role"]=="assistant")
    out.append({"id":r.get("id"),"user":u,"assistant":a})
print(json.dumps(out,ensure_ascii=False,indent=1)[:20000])
