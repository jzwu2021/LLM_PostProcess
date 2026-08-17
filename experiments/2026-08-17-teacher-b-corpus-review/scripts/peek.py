import json, sys
p = sys.argv[1]; a = int(sys.argv[2]); b = int(sys.argv[3])
L = open(p).readlines()
print("TOTAL", len(L))
for i in range(a, min(b, len(L))):
    d = json.loads(L[i])
    print("IDX", i, "KEYS", list(d.keys()))
    print(json.dumps(d, ensure_ascii=False)[:1600])
    print("====")
