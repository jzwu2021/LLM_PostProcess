import json, sys
path, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for i, l in enumerate(open(path), 1):
    if lo <= i <= hi:
        d = json.loads(l)
        print("===LINE", i, "KEYS", list(d.keys()))
        print(json.dumps(d, ensure_ascii=False)[:2200])
    if i > hi:
        break
