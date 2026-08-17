import json, sys
path, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path) as f:
    for i, line in enumerate(f):
        if lo <= i < hi:
            r = json.loads(line)
            print(json.dumps(r, ensure_ascii=False))
        if i >= hi:
            break
