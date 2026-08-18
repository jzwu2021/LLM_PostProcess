import json, sys
start = int(sys.argv[1]); n = int(sys.argv[2]); path = sys.argv[3]
recs = [json.loads(l) for l in open(path) if l.strip()][start:start+n]
for r in recs:
    print("=== ID:", r.get("id"), "| cat:", r.get("category"), "| type:", r.get("task_type"), "| diff:", r.get("difficulty"))
    print("concepts:", r.get("concepts"))
    for m in r.get("messages", []):
        print("--", m.get("role"), "--")
        print(m.get("content"))
    print()
