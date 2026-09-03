"""Self-test the holdout's ground truth before any model is scored.

An evaluation whose reference answers are wrong will confidently rank models on
noise. This checks two things:

  * every numeric answer is reproduced by an independent recomputation written
    from the question text rather than from the builder's helper;
  * every code item's hidden tests pass against a correct reference solution, so
    a model that solves the task as stated cannot be failed by a broken test.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ITEMS = HERE / "data/private_holdout_v1.jsonl"
HARNESS = (HERE / "data/code_harness.py").read_text()

GIB = 1024 ** 3
fail = []


def chk(cond, msg):
    if not cond:
        fail.append(msg)


# --------------------------------------------------------------------------- #
# independent recomputation, parsed out of the question text
# --------------------------------------------------------------------------- #

def recompute(item):
    q = item["messages"][1]["content"]
    n = [float(x) for x in re.findall(r"-?\d+\.?\d*", q.replace(",", ""))]
    k = item["quantity"]

    if k == "kv_bytes_per_token":
        layers, kvh, hd, db = n[0], n[1], n[2], n[3]
        return 2 * layers * kvh * hd * db
    if k == "kv_request_gib":
        layers, kvh, hd, db, ctx = n[0], n[1], n[2], n[3], n[4]
        return 2 * layers * kvh * hd * db * ctx / GIB
    if k == "weight_gib_per_device":
        params, db, tp = n[0], n[1], n[2]
        return params * 1e9 * db / tp / GIB
    if k == "paged_blocks":
        block, seq = n[0], n[1]
        return math.ceil(seq / block)
    if k == "allreduce_bytes_per_token":
        tp, layers, hidden, db = n[0], n[1], n[2], n[3]
        return 2 * hidden * db * 2 * (tp - 1) / tp * layers
    if k == "bubble_fraction":
        stages, micro = n[0], n[1]
        return (stages - 1) / (micro + stages - 1)
    if k == "effective_batch":
        micro, ga, world = n[0], n[1], n[2]
        return micro * ga * world
    if k == "nearest_rank_index":
        # positional parsing is fragile here ("from 1.", "1-based" both match),
        # so pull both operands by their semantic markers instead
        samples = float(re.search(r"You have (\d+) latency samples", q).group(1))
        pct = float(re.search(r"\bp(\d+) value", q).group(1))
        return math.ceil(pct / 100 * samples)
    if k == "spec_decode_breakeven":
        r, kk = n[0], n[1]
        return r * kk + 1
    if k == "littles_law_concurrency":
        rps, lat = n[0], n[1]
        return rps * lat
    if k == "mig_slices":
        total, per = n[0], n[1]
        return total // per
    if k == "amdahl_max_speedup":
        f = n[0]
        return 1 / (1 - f)
    if k == "availability_product":
        deps, p = n[0], n[1]
        return p ** deps
    if k == "read_amplification":
        return n[0]
    return None


# --------------------------------------------------------------------------- #
# reference solutions for the code items
# --------------------------------------------------------------------------- #

REFERENCE = {
"kv_cache_bytes": '''
def kv_cache_bytes(layers, seq_len, kv_heads, head_dim, bytes_per_value):
    vals = (layers, seq_len, kv_heads, head_dim, bytes_per_value)
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError("all arguments must be positive integers")
    return 2 * layers * seq_len * kv_heads * head_dim * bytes_per_value
''',
"paged_blocks": '''
def paged_blocks(sequence_length, block_size):
    for v in (sequence_length, block_size):
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError("arguments must be positive integers")
    return (sequence_length + block_size - 1) // block_size
''',
"valid_tensor_parallel": '''
def valid_tensor_parallel(world_size, tp):
    for v in (world_size, tp):
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            return False
    return world_size % tp == 0
''',
"first_duplicate": '''
def first_duplicate(request_ids):
    seen = set()
    for rid in request_ids:
        if rid in seen:
            return rid
        seen.add(rid)
    return None
''',
"percentile_nearest_rank": '''
import math
def percentile_nearest_rank(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    idx = math.ceil(percentile / 100 * len(ordered))
    idx = max(1, min(idx, len(ordered)))
    return ordered[idx - 1]
''',
"parse_tool_call": '''
import json
def parse_tool_call(payload, allowed):
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) != {"name", "arguments"}:
        return None
    name, args = obj.get("name"), obj.get("arguments")
    if not isinstance(name, str) or name not in allowed:
        return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}
''',
"bounded_retry": '''
def bounded_retry(fn, attempts, is_retryable):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not is_retryable(exc) or i == attempts - 1:
                raise
    raise last
''',
"classify_phase": '''
def classify_phase(prompt_tokens, generated_tokens, threshold):
    if prompt_tokens < 0 or generated_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if generated_tokens == 0:
        return "prefill"
    return "prefill" if prompt_tokens / generated_tokens >= threshold else "decode"
''',
}


def run_code_item(item, solution):
    ns = {}
    try:
        exec(solution, ns)
        exec(HARNESS, ns)
    except Exception as exc:
        return [f"setup failed: {exc}"]
    problems = []
    for t in item["tests"]:
        try:
            got = eval(t["expr"], ns)
        except Exception as exc:
            problems.append(f"{t['expr']} raised {type(exc).__name__}: {exc}")
            continue
        exp = t["expected"]
        if isinstance(exp, list):
            exp = tuple(exp) if isinstance(got, tuple) else exp
        if got != exp:
            problems.append(f"{t['expr']} -> {got!r}, expected {exp!r}")
    return problems


def main():
    items = [json.loads(l) for l in ITEMS.open() if l.strip()]
    numeric = [i for i in items if i["kind"] == "numeric"]
    code = [i for i in items if i["kind"] == "code"]

    print(f"items: {len(items)} ({len(numeric)} numeric, {len(code)} code)")

    for it in numeric:
        got = recompute(it)
        chk(got is not None, f"{it['id']}: no independent recomputation for {it['quantity']}")
        if got is None:
            continue
        want = it["answer"]
        tol = max(abs(want) * max(it["tolerance_rel"], 1e-9), 1e-9)
        chk(abs(got - want) <= tol,
            f"{it['id']} ({it['quantity']}): stored {want}, recomputed {got}")

    for it in code:
        problems = run_code_item(it, REFERENCE[it["function"]])
        for p in problems:
            fail.append(f"{it['id']} ({it['function']}): reference solution fails {p}")

    if fail:
        print(f"\nSELFTEST_FAIL {len(fail)}")
        for f in fail[:20]:
            print(" -", f)
        return 1
    print("\nSELFTEST_PASS: every numeric answer reproduced independently, "
          "every hidden test passes against a correct reference solution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
