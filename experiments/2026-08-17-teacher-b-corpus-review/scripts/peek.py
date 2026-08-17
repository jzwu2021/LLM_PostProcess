import json, sys
path = sys.argv[1]; a = int(sys.argv[2]); b = int(sys.argv[3])
lines = open(path).read().splitlines()[a:b]
for i, l in enumerate(lines, a + 1):
    d = json.loads(l)
    print("=== LINE", i, "KEYS", list(d.keys()))
    print(json.dumps(d, ensure_ascii=False)[:2200])
    print()
