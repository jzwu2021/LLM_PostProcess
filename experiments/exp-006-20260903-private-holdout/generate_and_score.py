"""Generate answers for the private holdout and score them objectively.

Decoding is greedy and identical for every arm, so nothing in the comparison
depends on sampling. Scoring never inspects prose: a numeric item is right if the
parsed number matches the reference within tolerance, a code item is right if the
extracted function passes hidden tests.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ITEMS = HERE / "data/private_holdout_v1.jsonl"
HARNESS = (HERE / "data/code_harness.py").read_text()

MAX_TOKENS = 12288  # capped runs measured verbosity, not capability


def parse_number(text: str):
    """Last 'ANSWER: <number>' wins; fall back to the final bare number."""
    hits = re.findall(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if not hits:
        hits = re.findall(r"(-?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if not hits:
        return None
    try:
        return float(hits[-1].replace(",", ""))
    except ValueError:
        return None


def extract_code(text: str):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text


def score_numeric(item, output):
    got = parse_number(output)
    if got is None:
        return False, "no number parsed"
    want = item["answer"]
    tol = abs(want) * item["tolerance_rel"]
    if item["tolerance_rel"] == 0.0:
        ok = abs(got - want) < 1e-9
    else:
        ok = abs(got - want) <= max(tol, 1e-9)
    return ok, f"{got} vs {want}"


def score_code(item, output):
    ns = {}
    try:
        exec(extract_code(output), ns)
        exec(HARNESS, ns)
    except Exception as exc:
        return False, f"exec failed: {type(exc).__name__}: {exc}"
    passed = 0
    for t in item["tests"]:
        try:
            got = eval(t["expr"], ns)
        except Exception:
            continue
        exp = t["expected"]
        if isinstance(exp, list) and isinstance(got, tuple):
            exp = tuple(exp)
        if got == exp:
            passed += 1
    return passed == len(item["tests"]), f"{passed}/{len(item['tests'])} tests"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    items = [json.loads(l) for l in ITEMS.open() if l.strip()]
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=16384, gpu_memory_utilization=0.90,
              trust_remote_code=True)
    tok = llm.get_tokenizer()

    prompts = [tok.apply_chat_template(it["messages"], tokenize=False,
                                       add_generation_prompt=True) for it in items]
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outs = llm.generate(prompts, params)

    records, truncated = [], 0
    for it, o in zip(items, outs):
        text = o.outputs[0].text
        if o.outputs[0].finish_reason == "length":
            truncated += 1
        if it["kind"] == "numeric":
            ok, detail = score_numeric(it, text)
        else:
            ok, detail = score_code(it, text)
        records.append({"id": it["id"], "kind": it["kind"],
                        "in_v04_scope": it["in_v04_scope"], "correct": bool(ok),
                        "detail": detail, "finish_reason": o.outputs[0].finish_reason,
                        "output": text})

    n = len(records)
    correct = sum(r["correct"] for r in records)
    summary = {
        "model": args.label, "path": args.model, "items": n,
        "correct": correct, "accuracy": correct / n,
        "truncated": truncated,
        "by_kind": {}, "by_scope": {},
    }
    for k in ("numeric", "code"):
        sub = [r for r in records if r["kind"] == k]
        if sub:
            summary["by_kind"][k] = {"n": len(sub),
                                     "correct": sum(r["correct"] for r in sub),
                                     "accuracy": sum(r["correct"] for r in sub) / len(sub)}
    for scope, name in ((True, "in_v04_scope"), (False, "out_of_v04_scope")):
        sub = [r for r in records if r["in_v04_scope"] is scope]
        if sub:
            summary["by_scope"][name] = {"n": len(sub),
                                         "correct": sum(r["correct"] for r in sub),
                                         "accuracy": sum(r["correct"] for r in sub) / len(sub)}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"WROTE {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
