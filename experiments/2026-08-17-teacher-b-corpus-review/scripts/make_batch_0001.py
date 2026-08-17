#!/usr/bin/env python3
"""Build teacher-B provisional BLIND review batch train-batch-0001.jsonl.

Reads ONLY research/ai-infra-expert/corpus/train.jsonl (source_user/source_assistant).
Corrected answers are authored independently by the reviewing model (claude-opus-5-current).
"""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0001.jsonl")
START, COUNT = 0, 10

DEF_KV = (
"Mechanism. In a decoder-only transformer, every attention layer projects each token into query, key and value "
"vectors. Because decoding is autoregressive and causal, the keys and values of tokens 1..t-1 are identical at "
"step t to what they were when those tokens were first processed. The KV cache stores those per-layer K and V "
"tensors in GPU HBM so step t only computes Q/K/V for the single new token and attends against the cached "
"history. This turns per-step attention compute from O(t^2) recomputation of the whole prefix into O(t) work "
"against cached state, which is why it matters: it is the difference between decode latency growing quadratically "
"with context and growing roughly linearly.\n\n"
"Size model. bytes = 2 (K and V) * num_layers * seq_len * num_kv_heads * head_dim * dtype_bytes, per sequence. "
"Note num_kv_heads, not num_query_heads: with GQA/MQA the cache shrinks by the query:kv head ratio. Example, "
"order-of-magnitude only: 40 layers, 8 KV heads, head_dim 128, fp16 (2 B) at 8k tokens is "
"2*40*8192*8*128*2 B ~= 1.34 GiB per sequence. Verify against the actual config.json before trusting this number.\n\n"
"Boundary condition. The cache is a memory-for-compute trade, so it stops helping once it stops fitting. KV cache "
"grows linearly with sequence length AND with concurrency, so on a fixed HBM budget it directly caps max batch "
"size: usable_kv_bytes = HBM - weights - activations - fragmentation/allocator reserve. Past that point the "
"server either preempts/evicts sequences, swaps to host memory over PCIe, or OOMs. It also gives no benefit for "
"prefill (all prefix tokens are computed once anyway) and no benefit for a single-token, non-autoregressive "
"forward pass. A second boundary: any change that invalidates the prefix (edited system prompt, different "
"sampling branch that rewinds tokens) invalidates the cached entries for that suffix.\n\n"
"Falsifiable claim. If you disable the KV cache on a fixed model and prompt, per-token decode latency should grow "
"visibly with position and end-to-end decode throughput should fall by a large factor, while peak HBM use drops.\n\n"
"Evidence to collect. config.json (num_hidden_layers, num_key_value_heads, head_dim, torch_dtype); the server's "
"reported KV block/page count and utilization (e.g. vLLM gpu_memory_utilization and cache block stats); "
"nvidia-smi or torch.cuda.memory_allocated at steady state; a latency-vs-position curve with and without caching."
)

CONTRAST = (
"Direct contrast. Naive (no cache): at decode step t the model re-runs the full forward pass over all t tokens, "
"recomputing K and V for the entire prefix; attention cost per step is O(t) per layer per head and total decode "
"cost over N tokens is O(N^2). Cached: K and V for tokens 1..t-1 are read from HBM, only the new token's Q/K/V "
"are computed, and the new K/V are appended; per-step cost is O(t) attention reads but O(1) projection work, and "
"the pass is memory-bandwidth-bound rather than FLOP-bound.\n\n"
"Concrete mechanism. Per layer the cache holds two tensors shaped roughly "
"[batch, num_kv_heads, seq_len, head_dim]; at step t the runtime writes the new K/V into slot t (in vLLM/paged "
"implementations, into a fixed-size block from a block table rather than one contiguous buffer) and the attention "
"kernel reads slots 0..t. Naive decoding instead materialises the whole [t, hidden] activation stack every step, "
"so it burns FLOPs and transient activation memory it will immediately discard.\n\n"
"What actually changes. Compute per generated token drops by roughly a factor of t; steady-state HBM rises by "
"2 * layers * seq * kv_heads * head_dim * dtype_bytes per sequence; the bottleneck moves from GEMM throughput to "
"HBM bandwidth and to allocator/paging behaviour. Consequently the naive path is FLOP-limited and the cached path "
"is capacity- and bandwidth-limited, which is why long-context serving is a memory problem, not a math problem.\n\n"
"Boundary condition. The cached path is not universally better. For very short generations, or for pure prefill / "
"single forward passes / scoring and classification workloads, the cache buys almost nothing and only costs "
"memory. Under high concurrency with long contexts the cache is what forces preemption, recompute-on-evict, or "
"host-memory swap over PCIe, and at that point a recompute (naive-like) fallback can actually be the cheaper "
"policy. So the correct statement is: caching trades HBM capacity for decode compute, and it wins exactly while "
"the working set fits.\n\n"
"Falsifiable claim. Sweep generation length at fixed prompt: no-cache total decode time should scale "
"super-linearly with output length while cached decode time scales near-linearly, and peak HBM should move the "
"other way.\n\n"
"Evidence to collect. Per-token latency vs. position for both modes; nsys/torch profiler trace showing whether "
"time sits in projection GEMMs (naive) or attention/HBM reads (cached); server cache utilization and preemption "
"counters; model config for the size formula."
)

FAILURE = (
"Failure mode 1: capacity exhaustion and preemption under concurrency. The cache grows linearly in both sequence "
"length and number of concurrent sequences, so admitted load times context length is what fills HBM, not request "
"rate alone. Mechanism: usable_kv_bytes = HBM - weights - activations - allocator reserve; when the scheduler "
"cannot allocate a block for the next token it must preempt, evict-and-recompute, or swap to host memory over "
"PCIe. Symptom: p99 TPOT and TTFT spike non-linearly and throughput collapses while GPU SM utilization stays "
"moderate, because the system is capacity-bound rather than compute-bound. Boundary condition: this appears only "
"once total KV working set approaches the budget; below that, adding concurrency improves throughput.\n\n"
"Failure mode 2: fragmentation and over-reservation versus quantization/eviction trade-offs. Contiguous "
"per-sequence allocation sized to max_seq_len wastes memory for short requests; paged/block allocation fixes most "
"of that but introduces block-table overhead, a tail of internal fragmentation in the last partial block, and "
"cross-request reuse only when prefixes match exactly. The usual mitigations each carry a real cost: GQA/MQA cuts "
"the cache by the query:kv head ratio but is a model architecture decision, not a serving knob; KV quantization to "
"fp8/int8 roughly halves or quarters the cache but perturbs attention numerics and must be quality-gated; "
"offloading to host memory is bounded by PCIe bandwidth and adds per-token stalls; sliding-window or eviction "
"policies bound memory but silently drop long-range context and change output.\n\n"
"Falsifiable claim. Ramping concurrency at fixed context should show throughput rising then falling sharply at "
"the point where cache utilization approaches 100% and preemption counters become non-zero; the knee should move "
"right if you enable fp8 KV or shorten max context, and left if you lengthen context.\n\n"
"Evidence to collect. Cache utilization and preemption/recompute counters over time; peak and steady-state HBM; "
"p50/p95/p99 TTFT and TPOT versus concurrency; before/after task-quality scores when enabling KV quantization or "
"windowed attention; model config for the size model.\n\n"
"Rollback gate. Treat any memory-saving change as reverted-by-default if accuracy on a held-out task set regresses "
"beyond an agreed margin, or if preemption/recompute rate or p99 TPOT worsens versus the recorded baseline."
)

ANSWERS = {
    "corpus-00001": DEF_KV, "corpus-00003": DEF_KV, "corpus-00004": DEF_KV, "corpus-00005": DEF_KV,
    "corpus-00006": CONTRAST, "corpus-00007": CONTRAST, "corpus-00008": CONTRAST,
    "corpus-00009": CONTRAST, "corpus-00010": CONTRAST,
    "corpus-00012": FAILURE,
}

RISKS = [
    "Source answer is a single generic sentence reused across distinct prompts, so it does not answer contrast or failure-mode variants.",
    "No quantitative size model, no units, and no explicit boundary condition tied to HBM budget or concurrency.",
    "No falsifiable claim, no required evidence, and no rollback gate, so it cannot be operationally validated.",
]
EVIDENCE = [
    "model config.json: num_hidden_layers, num_key_value_heads, head_dim, torch_dtype",
    "serving-engine KV cache block count, utilization and preemption/recompute counters",
    "nvidia-smi / torch.cuda memory at steady state and peak HBM",
    "per-token decode latency vs. sequence position and vs. concurrency (p50/p95/p99 TTFT, TPOT)",
]

def main():
    rows = []
    with open(CORPUS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < START:
                continue
            if i >= START + COUNT:
                break
            d = json.loads(line)
            u = next(m["content"] for m in d["messages"] if m["role"] == "user")
            a = next(m["content"] for m in d["messages"] if m["role"] == "assistant")
            sid = d["id"]
            assert sid in ANSWERS, sid
            rows.append({
                "source_id": sid,
                "teacher_lane": "teacher-B",
                "teacher_model": "claude-opus-5-current",
                "calibration_status": "provisional",
                "decision": "rewrite",
                "source_user": u,
                "source_assistant": a,
                "corrected_answer": ANSWERS[sid],
                "quality_dimensions": {
                    "technical_correctness": 3,
                    "instruction_coverage": 1 if sid not in ("corpus-00001", "corpus-00003", "corpus-00004", "corpus-00005") else 2,
                    "operational_safety": 3,
                },
                "risks": RISKS,
                "evidence_required": EVIDENCE,
                "confidence": 0.82,
            })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(rows))

if __name__ == "__main__":
    main()
