import json, sys
start=int(sys.argv[1]); n=int(sys.argv[2])
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
with open(p) as f:
    for i,l in enumerate(f,1):
        if i<start: continue
        if i>=start+n: break
        d=json.loads(l)
        print("### line",i)
        print(json.dumps(d,ensure_ascii=False)[:4000])
