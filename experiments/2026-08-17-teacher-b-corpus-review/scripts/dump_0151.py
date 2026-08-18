import json
for i, line in enumerate(open('/tmp/tb_src.jsonl')):
    r = json.loads(line)
    u = a = ''
    for m in r['messages']:
        if m['role'] == 'user': u = m['content']
        elif m['role'] == 'assistant': a = m['content']
    print('===', i, r.get('id'))
    print('--U--'); print(u)
    print('--A(len %d)--' % len(a)); print(a[:1200])
