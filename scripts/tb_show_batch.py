import json, sys
start=int(sys.argv[1]); n=int(sys.argv[2])
p="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
with open(p) as f:
    for i, line in enumerate(f):
        if i < start-1: continue
        if i >= start-1+n: break
        d=json.loads(line)
        print("IDX", i+1, "ID", d.get("id"))
        print("KEYS", list(d.keys()))
        for k in d:
            if k!="id":
                print("<<%s>>"%k, str(d[k])[:1800])
        print("="*90)
