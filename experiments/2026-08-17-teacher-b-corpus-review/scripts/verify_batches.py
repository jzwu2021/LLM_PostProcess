import json, glob, sys, os, re

D = 'experiments/2026-08-17-teacher-b-corpus-review/results'
errs = []

def load_corpus(p):
    out = []
    for line in open(p):
        d = json.loads(line)
        m = {x['role']: x['content'] for x in d['messages']}
        out.append((d['id'], m['user'], m['assistant']))
    return out

corp = {'train': load_corpus('research/ai-infra-expert/corpus/train.jsonl'),
        'validation': load_corpus('research/ai-infra-expert/corpus/validation.jsonl')}

REQ = ["source_id","teacher_lane","teacher_model","calibration_status","decision","source_user",
       "source_assistant","corrected_answer","quality_dimensions","risks","evidence_required","confidence"]
seen = set()
seqs = {'train': [], 'validation': []}

for split in ('train','validation'):
    files = sorted(glob.glob(f'{D}/{split}-batch-*.jsonl'))
    for fp in files:
        raw = open(fp, 'rb').read().decode('utf-8')
        lines = [l for l in raw.split('\n') if l.strip()]
        if not raw.endswith('\n'):
            errs.append(f'{fp}: missing trailing newline')
        if len(lines) != 10:
            errs.append(f'{fp}: expected 10 records, got {len(lines)}')
        for ln, l in enumerate(lines, 1):
            try:
                r = json.loads(l)
            except Exception as e:
                errs.append(f'{fp}:{ln} JSON parse: {e}'); continue
            for k in REQ:
                if k not in r: errs.append(f'{fp}:{ln} missing field {k}')
            if len(r) != 12: errs.append(f'{fp}:{ln} field count {len(r)} != 12')
            if r.get('teacher_lane') != 'teacher-B': errs.append(f'{fp}:{ln} bad teacher_lane')
            if r.get('teacher_model') != 'claude-opus-5-current': errs.append(f'{fp}:{ln} bad teacher_model')
            if r.get('calibration_status') != 'provisional': errs.append(f'{fp}:{ln} bad calibration_status')
            if r.get('decision') not in ('keep','rewrite','reject'): errs.append(f'{fp}:{ln} bad decision')
            ca = r.get('corrected_answer')
            if not isinstance(ca, str) or not ca.strip(): errs.append(f'{fp}:{ln} empty corrected_answer')
            c = r.get('confidence')
            if not isinstance(c,(int,float)) or not (0.0 <= c <= 1.0): errs.append(f'{fp}:{ln} bad confidence')
            qd = r.get('quality_dimensions')
            if not isinstance(qd, dict): errs.append(f'{fp}:{ln} quality_dimensions not object')
            else:
                for k in ('technical_correctness','instruction_coverage','operational_safety'):
                    v = qd.get(k)
                    if not isinstance(v,int) or isinstance(v,bool) or not (1 <= v <= 5):
                        errs.append(f'{fp}:{ln} bad quality_dimensions.{k}')
            for k in ('risks','evidence_required'):
                v = r.get(k)
                if not isinstance(v, list) or not all(isinstance(x,str) for x in v):
                    errs.append(f'{fp}:{ln} {k} not string array')
            sid = r.get('source_id')
            if sid in seen: errs.append(f'{fp}:{ln} duplicate source_id {sid}')
            seen.add(sid)
            seqs[split].append((sid, r.get('source_user'), r.get('source_assistant')))

    n = len(seqs[split])
    pref = corp[split][:n]
    if seqs[split] != pref:
        for i,(g,e) in enumerate(zip(seqs[split], pref)):
            if g != e:
                errs.append(f'{split}[{i}] prefix mismatch: got id={g[0]} expected id={e[0]}'
                            f' user_eq={g[1]==e[1]} asst_eq={g[2]==e[2]}')
                break
        if n > len(corp[split]): errs.append(f'{split}: more records than corpus')

print(f"train={len(seqs['train'])}/5399 validation={len(seqs['validation'])}/601 total={len(seen)}/6000")
if errs:
    print('VERIFY=FAIL')
    for e in errs[:40]: print(' ', e)
    sys.exit(1)
print('VERIFY=PASS')
