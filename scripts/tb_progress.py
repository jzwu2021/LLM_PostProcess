import json, os, glob
base="experiments/2026-08-17-teacher-b-corpus-review/results"
tr=0; va=0; ids=set()
for f in sorted(glob.glob(base+"/train-batch-*.jsonl")):
    n=sum(1 for l in open(f) if l.strip()); tr+=n
for f in sorted(glob.glob(base+"/validation-batch-*.jsonl")):
    n=sum(1 for l in open(f) if l.strip()); va+=n
print("train",tr,"validation",va,"total",tr+va)
nb_t=len(glob.glob(base+"/train-batch-*.jsonl")); nb_v=len(glob.glob(base+"/validation-batch-*.jsonl"))
print("batches",nb_t,nb_v)
