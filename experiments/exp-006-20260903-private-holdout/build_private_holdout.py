"""Build a private, objectively-scored holdout for capability comparison.

WHAT THIS DOES AND DOES NOT SOLVE
---------------------------------
Contamination has two distinct routes into an evaluation:

  (a) answer-level: the evaluator knows the expected answer text, and training
      data authored by the same person carries that text's content;
  (b) topic-level: the evaluator chose subjects the training data happens to
      cover.

This set eliminates (a) and does not eliminate (b).

Every item's ground truth is computed by a reference function from randomly
sampled parameters, or decided by executing hidden unit tests. There is no
expected prose, so no prose can leak. The model must perform the arithmetic or
write working code.

Route (b) remains and is declared: the same author wrote the v0.4 corpus and
selected these subjects. To bound it, the set deliberately includes quantities
that no v0.4 mechanism covers (speculative decoding break-even, Little's law,
MIG partitioning, Amdahl bound), marked `in_v04_scope: false`.

Parameters are drawn from a fixed seed so the set is reproducible, and the file
is hash-pinned on write.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random

OUT = pathlib.Path(__file__).resolve().parent / "data"
SEED = 20260903
GIB = 1024 ** 3


# --------------------------------------------------------------------------- #
# reference implementations: these define ground truth, the model never sees them
# --------------------------------------------------------------------------- #

def ref_kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes):
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def ref_kv_request_gib(layers, kv_heads, head_dim, dtype_bytes, ctx):
    return ref_kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes) * ctx / GIB


def ref_weight_gib_per_device(params_b, dtype_bytes, tp):
    return params_b * 1e9 * dtype_bytes / tp / GIB


def ref_paged_blocks(seq_len, block_size):
    return (seq_len + block_size - 1) // block_size


def ref_allreduce_bytes_per_token(hidden, dtype_bytes, layers, tp):
    per_layer = 2 * hidden * dtype_bytes * 2 * (tp - 1) / tp
    return per_layer * layers


def ref_bubble_fraction(stages, microbatches):
    return (stages - 1) / (microbatches + stages - 1)


def ref_effective_batch(micro, grad_accum, world):
    return micro * grad_accum * world


def ref_nearest_rank_index(n, percentile):
    return math.ceil(percentile / 100 * n)


def ref_spec_breakeven(draft_cost_ratio, k):
    """Minimum accepted tokens per step for speculation to beat plain decoding.

    Plain decoding produces 1 token per target forward. A speculative step costs
    k draft passes plus one target forward, so cost = k*r + 1 in units of the
    target forward. Break-even accepted tokens a satisfies a >= k*r + 1.
    """
    return draft_cost_ratio * k + 1


def ref_littles_law_concurrency(throughput_rps, latency_s):
    return throughput_rps * latency_s


def ref_mig_slices(total_slices, per_instance):
    return total_slices // per_instance


def ref_amdahl_max_speedup(fraction_optimised):
    return 1 / (1 - fraction_optimised)


def ref_availability_product(p, n):
    return p ** n


def ref_read_amplification(tp):
    return float(tp)


# --------------------------------------------------------------------------- #
# numeric item builders
# --------------------------------------------------------------------------- #

SYSTEM = ("You are an AI/LLM infrastructure engineer. Give the final numeric answer "
          "on the last line in the exact form 'ANSWER: <number>' with no units and no "
          "thousands separators. Show your working before that line.")


def numeric_items(rng):
    items = []

    def add(key, prompt, value, tol, in_scope, unit):
        items.append({
            "id": f"priv-num-{len(items) + 1:03d}",
            "kind": "numeric",
            "quantity": key,
            "in_v04_scope": in_scope,
            "unit": unit,
            "tolerance_rel": tol,
            "answer": value,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
        })

    for _ in range(6):
        layers = rng.choice([28, 32, 40, 48, 60, 80])
        kv_heads = rng.choice([4, 8, 16])
        head_dim = rng.choice([64, 128])
        db = rng.choice([1, 2])
        add("kv_bytes_per_token",
            f"A transformer has {layers} layers, {kv_heads} key-value heads and a head "
            f"dimension of {head_dim}. The KV cache stores {db} byte(s) per value. "
            f"How many bytes of KV cache does one token occupy across all layers?",
            float(ref_kv_bytes_per_token(layers, kv_heads, head_dim, db)), 0.001, True, "bytes")

    for _ in range(5):
        layers = rng.choice([32, 48, 80])
        kv_heads = rng.choice([8, 16])
        head_dim = 128
        db = 2
        ctx = rng.choice([4096, 8192, 16384, 32768])
        add("kv_request_gib",
            f"A model has {layers} layers, {kv_heads} key-value heads, head dimension "
            f"{head_dim}, and a 2-byte KV cache. A request holds a full context of "
            f"{ctx} tokens. How many GiB of KV cache does that single request occupy? "
            f"Use 1 GiB = 1073741824 bytes.",
            ref_kv_request_gib(layers, kv_heads, head_dim, db, ctx), 0.01, True, "GiB")

    for _ in range(5):
        params = rng.choice([7, 9, 13, 32, 70])
        db = rng.choice([1, 2])
        tp = rng.choice([1, 2, 4, 8])
        add("weight_gib_per_device",
            f"A {params}B-parameter model is served at {db} byte(s) per parameter with "
            f"tensor parallel degree {tp}. How many GiB of weights does each device hold? "
            f"Use 1e9 for the billion multiplier and 1 GiB = 1073741824 bytes.",
            ref_weight_gib_per_device(params, db, tp), 0.01, True, "GiB")

    for _ in range(5):
        seq = rng.randint(1, 40000)
        block = rng.choice([8, 16, 32, 64])
        add("paged_blocks",
            f"A paged KV cache allocates whole blocks of {block} tokens. A sequence is "
            f"{seq} tokens long. How many blocks are required?",
            float(ref_paged_blocks(seq, block)), 0.0, True, "blocks")

    for _ in range(4):
        hidden = rng.choice([2048, 4096, 5120, 8192])
        db = 2
        layers = rng.choice([32, 48, 80])
        tp = rng.choice([2, 4, 8])
        add("allreduce_bytes_per_token",
            f"A tensor-parallel group of size {tp} runs a model with {layers} layers and "
            f"hidden size {hidden} at {db} bytes per element. Each layer performs two "
            f"ring all-reduces per token over a hidden-size activation, and a ring "
            f"all-reduce moves 2(N-1)/N of the payload per rank. How many bytes does one "
            f"rank move per generated token across all layers?",
            ref_allreduce_bytes_per_token(hidden, db, layers, tp), 0.01, True, "bytes")

    for _ in range(4):
        stages = rng.choice([2, 4, 8])
        micro = rng.choice([4, 8, 16, 32])
        add("bubble_fraction",
            f"A pipeline has {stages} stages and keeps {micro} microbatches in flight. "
            f"Using bubble fraction = (stages - 1) / (microbatches + stages - 1), what "
            f"fraction of stage time is idle? Give a decimal fraction.",
            ref_bubble_fraction(stages, micro), 0.01, True, "fraction")

    for _ in range(4):
        micro = rng.choice([1, 2, 4, 8])
        ga = rng.choice([1, 2, 4, 8])
        world = rng.choice([1, 2, 4, 8, 16])
        add("effective_batch",
            f"Training uses a per-device micro-batch of {micro}, gradient accumulation "
            f"{ga}, across {world} ranks. What is the effective batch size in sequences "
            f"per optimiser step?",
            float(ref_effective_batch(micro, ga, world)), 0.0, True, "sequences")

    for _ in range(4):
        n = rng.choice([37, 100, 250, 999])
        p = rng.choice([50, 90, 95, 99])
        add("nearest_rank_index",
            f"You have {n} latency samples sorted ascending, indexed from 1. Using the "
            f"nearest-rank definition, index = ceil(P/100 * N), which 1-based index holds "
            f"the p{p} value?",
            float(ref_nearest_rank_index(n, p)), 0.0, True, "index")

    # deliberately outside every v0.4 mechanism
    for _ in range(4):
        r = rng.choice([0.1, 0.2, 0.25, 0.4])
        k = rng.choice([2, 3, 4, 5])
        add("spec_decode_breakeven",
            f"In speculative decoding a draft pass costs {r} of one target forward. A "
            f"step proposes {k} tokens, costing {k} draft passes plus one target forward. "
            f"Plain decoding produces one token per target forward. How many accepted "
            f"tokens per step are needed for speculation to break even?",
            ref_spec_breakeven(r, k), 0.01, False, "tokens")

    for _ in range(3):
        rps = rng.choice([12, 40, 125, 300])
        lat = rng.choice([0.4, 1.5, 2.0, 3.2])
        add("littles_law_concurrency",
            f"A service sustains {rps} requests per second at a mean end-to-end latency of "
            f"{lat} seconds. By Little's law, what is the mean number of requests in the "
            f"system concurrently?",
            ref_littles_law_concurrency(rps, lat), 0.01, False, "requests")

    for _ in range(3):
        total = rng.choice([7, 8])
        per = rng.choice([1, 2, 3, 4])
        add("mig_slices",
            f"A GPU exposes {total} compute slices. Each MIG instance is configured with "
            f"{per} slice(s). How many whole instances fit?",
            float(ref_mig_slices(total, per)), 0.0, False, "instances")

    for _ in range(3):
        f = rng.choice([0.5, 0.75, 0.9, 0.95])
        add("amdahl_max_speedup",
            f"A stage accounts for {f} of total runtime. If that stage were made "
            f"infinitely fast, what is the maximum achievable end-to-end speedup?",
            ref_amdahl_max_speedup(f), 0.01, False, "x")

    for _ in range(3):
        p = rng.choice([0.99, 0.995, 0.999])
        n = rng.choice([3, 5, 8])
        add("availability_product",
            f"A request requires {n} independent dependencies, each available {p} of the "
            f"time. What is the combined availability, assuming independence? Give a "
            f"decimal fraction to at least five places.",
            ref_availability_product(p, n), 0.0005, True, "fraction")

    for _ in range(2):
        tp = rng.choice([2, 4, 8])
        add("read_amplification",
            f"A model artifact is not shardable, so each of {tp} tensor-parallel ranks "
            f"reads the entire file and keeps only its own shard. By what factor does the "
            f"storage source serve more bytes than the artifact size?",
            ref_read_amplification(tp), 0.0, True, "x")

    return items


# --------------------------------------------------------------------------- #
# code items: ground truth is the hidden test, never prose
# --------------------------------------------------------------------------- #

CODE_SYSTEM = ("You are an AI/LLM infrastructure engineer. Return only a Python function "
               "in a single ```python code block. No explanation outside the block.")

CODE_SPECS = [
    dict(
        name="kv_cache_bytes", in_scope=True,
        prompt=("Write `kv_cache_bytes(layers, seq_len, kv_heads, head_dim, bytes_per_value)` "
                "returning the exact integer KV cache size in bytes for one sequence, "
                "counting both the key and the value tensor. Raise ValueError for any "
                "argument that is not a positive integer. The result must be an int."),
        tests=[
            ("kv_cache_bytes(2, 4, 2, 8, 2)", 2 * 2 * 4 * 2 * 8 * 2),
            ("kv_cache_bytes(48, 8192, 8, 128, 2)", 2 * 48 * 8192 * 8 * 128 * 2),
            ("type(kv_cache_bytes(4, 4, 4, 4, 1)) is int", True),
            ("_raises(lambda: kv_cache_bytes(0, 4, 4, 4, 1), ValueError)", True),
            ("_raises(lambda: kv_cache_bytes(4, -1, 4, 4, 1), ValueError)", True),
        ]),
    dict(
        name="paged_blocks", in_scope=True,
        prompt=("Write `paged_blocks(sequence_length, block_size)` returning the number of "
                "whole blocks needed, using integer arithmetic only, no floating point "
                "division. Raise ValueError if either argument is not a positive integer."),
        tests=[
            ("paged_blocks(1, 16)", 1),
            ("paged_blocks(16, 16)", 1),
            ("paged_blocks(17, 16)", 2),
            ("paged_blocks(10**15, 16)", (10**15 + 15) // 16),
            ("_raises(lambda: paged_blocks(10, 0), ValueError)", True),
            ("_raises(lambda: paged_blocks(0, 16), ValueError)", True),
        ]),
    dict(
        name="valid_tensor_parallel", in_scope=True,
        prompt=("Write `valid_tensor_parallel(world_size, tp)` returning True only when both "
                "are positive integers and tp divides world_size exactly. Return False "
                "otherwise. Do not raise for non-positive values, and reject booleans."),
        tests=[
            ("valid_tensor_parallel(8, 4)", True),
            ("valid_tensor_parallel(8, 3)", False),
            ("valid_tensor_parallel(0, 1)", False),
            ("valid_tensor_parallel(8, 0)", False),
            ("valid_tensor_parallel(-8, 4)", False),
            ("valid_tensor_parallel(True, 1)", False),
        ]),
    dict(
        name="first_duplicate", in_scope=False,
        prompt=("Write `first_duplicate(request_ids)` returning the first identifier that "
                "repeats in input order, or None when there is no repeat. Preserve input "
                "order; do not sort and do not rely on set iteration order."),
        tests=[
            ("first_duplicate(['a','b','c','b','a'])", "b"),
            ("first_duplicate(['x','y','z'])", None),
            ("first_duplicate([])", None),
            ("first_duplicate(['q','q'])", "q"),
            ("first_duplicate(list('abcdefghijb'))", "b"),
        ]),
    dict(
        name="percentile_nearest_rank", in_scope=True,
        prompt=("Write `percentile_nearest_rank(values, percentile)` returning the value at "
                "the nearest-rank position, index = ceil(P/100 * N) over the ascending sort, "
                "1-based. Return None for an empty input. Do not mutate the caller's list."),
        tests=[
            ("percentile_nearest_rank([1,2,3,4,5], 50)", 3),
            ("percentile_nearest_rank([1,2,3,4,5], 100)", 5),
            ("percentile_nearest_rank([5,1,3], 50)", 3),
            ("percentile_nearest_rank([], 50)", None),
            ("percentile_nearest_rank([7], 99)", 7),
            ("(lambda v: (percentile_nearest_rank(v, 50), v))([3,1,2])", (2, [3,1,2])),
        ]),
    dict(
        name="parse_tool_call", in_scope=True,
        prompt=("Write `parse_tool_call(payload, allowed)` taking a JSON string and a set of "
                "allowed tool names. Return a dict with keys 'name' and 'arguments' only "
                "when the payload is a JSON object whose 'name' is a string in `allowed` and "
                "whose 'arguments' is an object, and which has no other top-level keys. "
                "Return None in every other case, including invalid JSON. Never raise."),
        tests=[
            ("""parse_tool_call('{"name":"f","arguments":{"a":1}}', {"f"})""",
             {"name": "f", "arguments": {"a": 1}}),
            ("""parse_tool_call('{"name":"g","arguments":{}}', {"f"})""", None),
            ("""parse_tool_call('{"name":"f","arguments":[]}', {"f"})""", None),
            ("""parse_tool_call('{"name":"f","arguments":{},"extra":1}', {"f"})""", None),
            ("""parse_tool_call('not json', {"f"})""", None),
            ("""parse_tool_call('[1,2]', {"f"})""", None),
            ("""parse_tool_call('{"name":123,"arguments":{}}', {"f"})""", None),
        ]),
    dict(
        name="bounded_retry", in_scope=True,
        prompt=("Write `bounded_retry(fn, attempts, is_retryable)` that calls `fn()` and "
                "returns its result. On an exception, retry only while attempts remain and "
                "`is_retryable(exc)` is true; otherwise re-raise. Total calls must never "
                "exceed `attempts`. Do not sleep."),
        tests=[
            ("_retry_ok()", (3, "ok")),
            ("_retry_exhausted()", 2),
            ("_retry_nonretryable()", 1),
        ]),
    dict(
        name="classify_phase", in_scope=True,
        prompt=("Write `classify_phase(prompt_tokens, generated_tokens, threshold)` returning "
                "'prefill' when generated_tokens is 0, otherwise 'prefill' if "
                "prompt_tokens / generated_tokens >= threshold and 'decode' if below. Raise "
                "ValueError on negative counts or a non-positive threshold."),
        tests=[
            ("classify_phase(100, 0, 4)", "prefill"),
            ("classify_phase(100, 10, 4)", "prefill"),
            ("classify_phase(10, 100, 4)", "decode"),
            ("classify_phase(40, 10, 4)", "prefill"),
            ("_raises(lambda: classify_phase(-1, 1, 4), ValueError)", True),
            ("_raises(lambda: classify_phase(1, 1, 0), ValueError)", True),
        ]),
]

HARNESS = '''
import json, math
def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False

def _retry_ok():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return "ok"
    return (len(calls), bounded_retry(fn, 5, lambda e: True)) if False else (
        (lambda r: (len(calls), r))(bounded_retry(fn, 5, lambda e: True)))

def _retry_exhausted():
    calls = []
    def fn():
        calls.append(1)
        raise RuntimeError("always")
    try:
        bounded_retry(fn, 2, lambda e: True)
    except Exception:
        pass
    return len(calls)

def _retry_nonretryable():
    calls = []
    def fn():
        calls.append(1)
        raise ValueError("fatal")
    try:
        bounded_retry(fn, 5, lambda e: not isinstance(e, ValueError))
    except Exception:
        pass
    return len(calls)
'''


def code_items():
    items = []
    for spec in CODE_SPECS:
        items.append({
            "id": f"priv-code-{len(items) + 1:03d}",
            "kind": "code",
            "function": spec["name"],
            "in_v04_scope": spec["in_scope"],
            "tests": [{"expr": e, "expected": v} for e, v in spec["tests"]],
            "messages": [{"role": "system", "content": CODE_SYSTEM},
                         {"role": "user", "content": spec["prompt"]}],
        })
    return items


def main():
    rng = random.Random(SEED)
    items = numeric_items(rng) + code_items()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "private_holdout_v1.jsonl"
    with path.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    (OUT / "code_harness.py").write_text(HARNESS)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    n_num = sum(1 for i in items if i["kind"] == "numeric")
    n_code = sum(1 for i in items if i["kind"] == "code")
    out_scope = sum(1 for i in items if not i["in_v04_scope"])
    print(f"wrote {len(items)} items: {n_num} numeric, {n_code} code")
    print(f"outside every v0.4 mechanism: {out_scope}")
    print(f"seed {SEED}")
    print(f"sha256 {digest}")
    (OUT / "private_holdout_v1.sha256").write_text(f"{digest}  private_holdout_v1.jsonl\n")


if __name__ == "__main__":
    main()
