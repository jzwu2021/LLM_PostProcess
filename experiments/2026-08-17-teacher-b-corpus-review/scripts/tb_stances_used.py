import json, glob, re
EXP="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
s=set()
for f in sorted(glob.glob(f"{EXP}/results/train-batch-*.jsonl")):
    for l in open(f):
        if not l.strip(): continue
        ca=json.loads(l)["corrected_answer"]
        m=re.match(r"Analytical stance under test: ([a-zA-Z0-9\-]+)", ca)
        if m: s.add(m.group(1))
print(len(s))
for x in sorted(s): print(x)
