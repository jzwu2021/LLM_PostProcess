import json, sys
p = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
start = int(sys.argv[1]); n = int(sys.argv[2])
with open(p) as f:
    for i, line in enumerate(f):
        if start <= i < start + n:
            o = json.loads(line)
            print("### IDX", i, "KEYS", list(o.keys()))
            print(json.dumps(o, ensure_ascii=False)[:4000])
            print()
        if i >= start + n:
            break
