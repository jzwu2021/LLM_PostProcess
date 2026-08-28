#!/usr/bin/env python3
import argparse, json, time, urllib.request, urllib.error
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an AI/LLM Infrastructure engineering assistant. "
    "Answer the user's technical question directly and rigorously. "
    "State assumptions, units, formulas, trade-offs, uncertainty, and validation steps when relevant. "
    "Do not invent measurements or undocumented system facts."
)


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def post_json(url, payload, timeout=300):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def wait_ready(base, model, timeout_s=600):
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            payload = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': 'Reply with exactly READY.'},
                    {'role': 'user', 'content': 'READY'}
                ],
                'temperature': 0.0,
                'max_tokens': 8,
                'chat_template_kwargs': {'enable_thinking': False},
            }
            resp = post_json(base.rstrip('/') + '/chat/completions', payload, timeout=60)
            txt = ((resp.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
            if txt.strip():
                return
        except Exception as e:
            last_err = repr(e)
            time.sleep(5)
    raise SystemExit(f'SERVER_NOT_READY last_error={last_err}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-tokens', type=int, default=768)
    args = ap.parse_args()

    rows = load_jsonl(args.benchmark)
    if args.limit > 0:
        rows = rows[:args.limit]

    wait_ready(args.base, args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8') as out:
        for idx, row in enumerate(rows, 1):
            payload = {
                'model': args.model,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': row['question']},
                ],
                'temperature': 0.0,
                'max_tokens': args.max_tokens,
                'chat_template_kwargs': {'enable_thinking': False},
            }
            started = time.time()
            item = {
                'index': idx,
                'id': row['id'],
                'category': row.get('category'),
                'difficulty': row.get('difficulty'),
                'topic': row.get('topic'),
                'question': row['question'],
                'verifier': row.get('verifier'),
                'request': payload,
            }
            try:
                resp = post_json(args.base.rstrip('/') + '/chat/completions', payload)
                latency_ms = round((time.time() - started) * 1000, 3)
                choice = (resp.get('choices') or [{}])[0]
                msg = choice.get('message') or {}
                item.update({
                    'ok': True,
                    'retries_used': 0,
                    'latency_ms': latency_ms,
                    'finish_reason': choice.get('finish_reason'),
                    'response_content': msg.get('content') or '',
                })
            except Exception as e:
                latency_ms = round((time.time() - started) * 1000, 3)
                item.update({
                    'ok': False,
                    'retries_used': 0,
                    'latency_ms': latency_ms,
                    'finish_reason': None,
                    'response_content': '',
                    'error': repr(e),
                })
            out.write(json.dumps(item, ensure_ascii=False) + '\n')
            out.flush()
            print(f"{idx}/{len(rows)} {row['id']} {'OK' if item['ok'] else 'ERR'} finish={item['finish_reason']}", flush=True)
    print(f"GENERATIONS_DONE cases={len(rows)} output={args.output}")


if __name__ == '__main__':
    main()
