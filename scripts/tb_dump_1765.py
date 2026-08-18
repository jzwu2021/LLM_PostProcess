import json
out = []
for l in open('/tmp/tb_next10.jsonl'):
    d = json.loads(l)
    msgs = d['messages']
    u = [m for m in msgs if m['role'] == 'user'][0]['content']
    a = [m for m in msgs if m['role'] == 'assistant'][0]['content']
    out.append("=" * 30 + "\n" + d['id'] + " | cat=" + str(d.get('category')) + " | task=" + str(d.get('task_type')) + " | diff=" + str(d.get('difficulty')) + "\nconcepts=" + str(d.get('concepts')) + "\n--- USER ---\n" + u + "\n--- ASSISTANT ---\n" + a + "\n")
open('/tmp/tb_dump_1765.txt', 'w').write("\n".join(out))
print("ok", len(out))
