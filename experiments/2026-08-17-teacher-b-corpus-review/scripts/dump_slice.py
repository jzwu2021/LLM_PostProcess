import json, sys
start=int(sys.argv[1]); n=int(sys.argv[2])
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
out=[]
with open(p) as f:
    for i,l in enumerate(f):
        if i<start: continue
        if i>=start+n: break
        d=json.loads(l)
        out.append({"idx":i,"keys":list(d.keys()),"rec":d})
print(json.dumps(out,ensure_ascii=False,indent=1)[:20000])
