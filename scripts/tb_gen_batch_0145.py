import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0145.jsonl"
START, N = 1440, 10

OOM = """Assumptions: single-node vLLM/SGLang-class server, paged KV cache, FP16/BF16 weights, requests arriving concurrently with heterogeneous prompt lengths. "Intermittent" plus "after several concurrent requests" points at aggregate KV footprint, not a weight-sizing error.

Falsifiable hypothesis H1: OOM is driven by concurrent KV-cache demand exceeding the preallocated block pool plus residual free VRAM, i.e. peak_kv_bytes = sum over active seqs of 2 * n_layers * n_kv_heads * head_dim * dtype_bytes * ctx_len. Prediction: OOM onset correlates with the running sum of context lengths, not with request count, and is reproducible at a threshold value of that sum. If a fixed concurrency of short prompts never OOMs while the same concurrency of long prompts does at the same request rate, H1 survives; if OOM occurs at identical aggregate token counts regardless of layout, H1 is weakened in favor of fragmentation (H2).

Competing hypothesis H2: allocator fragmentation. Torch caching allocator reserved bytes stay high while allocated bytes are low, so a large contiguous activation request fails despite nominal headroom. Discriminator: torch.cuda.memory_reserved() - torch.cuda.memory_allocated() at failure time. A gap larger than the failing allocation implicates fragmentation. Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True should then move or remove the failure point; if it does not, H2 is falsified.

Competing hypothesis H3: activation/logits spike during prefill of a long prompt (chunked prefill disabled), where the transient logits tensor is vocab_size * batch_tokens * dtype_bytes. This scales with prefill chunk size, not with steady-state KV.

Controlled experiment: fix model, dtype, and server build. Sweep aggregate context tokens as the single independent variable via a closed-loop load generator at fixed concurrency C, prompt length L in {2k, 8k, 32k}, three repeats, randomized order. Record per-step torch.cuda.memory_allocated/reserved, gpu_cache_usage_perc, num_requests_running, num_requests_waiting, preemption counts, and nvidia-smi used memory sampled at 1 Hz. Success criterion: an OOM threshold predicted by the KV formula within 10 percent supports H1.

Confounders to control: other processes on the same GPU (check nvidia-smi pmon), CUDA graph capture memory, NCCL buffers if TP>1, fragmentation drift over long runs (restart between arms), speculative decoding or prefix-cache hits changing effective KV, and MIG or memory-limit settings.

Prioritized mitigations, cheapest and most reversible first:
1. Admission control: cap max_num_seqs and max_num_batched_tokens so worst-case KV is bounded by the block pool. Pure config, instant rollback.
2. Set gpu_memory_utilization explicitly (e.g. 0.90) so the block pool is preallocated and OOM becomes queueing/preemption instead of a hard crash.
3. Enable chunked prefill to bound the prefill activation and logits spike.
4. Enable expandable_segments to reduce fragmentation.
5. Enable prefix caching if the workload shares system prompts, which cuts KV for shared prefixes.
6. Reduce max_model_len to the true p99 context requirement rather than the model maximum.
7. KV quantization to FP8 (roughly halves KV bytes) only after a quality gate, since it is accuracy-affecting.
8. Tensor parallel or a larger GPU as the structural fix if the p99 aggregate context genuinely does not fit.

Evidence required before declaring root cause: an OOM traceback with the failing allocation size, allocated vs reserved at failure, the aggregate active context length at failure, and a reproduction at a predicted threshold.

Rollback gates: revert any change if p99 TTFT regresses more than 15 percent, throughput drops more than 10 percent, preemption rate exceeds 1 percent of requests, or (for FP8 KV) the eval suite moves beyond its pre-agreed noise band. Keep the previous config pinned and roll back within one deploy cycle."""

HANG = """Assumptions: PyTorch DDP/FSDP or a Megatron-style launcher over NCCL, job hangs before the first collective completes, no Python traceback, ranks alive but idle. This is a rendezvous or transport-establishment failure, not a numerical bug.

Falsifiable hypothesis H1: the hang is in NCCL transport setup, not in the store rendezvous, because one or more ranks cannot establish a usable path (wrong NIC selected, IB/RoCE not reachable, or peer-to-peer blocked). Prediction: NCCL_DEBUG=INFO shows all ranks reaching bootstrap and printing channel setup, then stalling with no "Connected all rings" line; and forcing NCCL_SOCKET_IFNAME to a known-good interface (plus NCCL_IB_DISABLE=1 as a control) unblocks it. If ranks never all print their bootstrap line, H1 is falsified in favor of H2.

Competing hypothesis H2: store/rendezvous failure — MASTER_ADDR/MASTER_PORT unreachable, mismatched WORLD_SIZE, duplicate RANK, or a straggler rank that never started. Discriminator: count distinct ranks that log init_process_group entry; missing ranks means H2.

Competing hypothesis H3: device visibility or affinity mismatch — two ranks bound to the same GPU via a bad CUDA_VISIBLE_DEVICES / LOCAL_RANK mapping, so a collective deadlocks. Discriminator: nvidia-smi shows fewer distinct processes than local ranks, or two PIDs on one GPU.

Ordered diagnosis plan:
1. Capture state: for every rank record hostname, RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR/PORT, CUDA_VISIBLE_DEVICES, NCCL and torch versions, driver version. Mismatch here explains most hangs.
2. Prove liveness: py-spy dump on each rank. Expect the stack in c10d ProcessGroupNCCL or in the TCPStore wait; that separates H1 from H2 immediately.
3. Turn on NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV and diff logs across ranks; the last common line localizes the stage.
4. Topology: nvidia-smi topo -m, ibstat / ibv_devinfo for IB or RoCE, confirm GID index and that the RoCE VLAN/PFC config matches on all hosts.
5. Minimal reproducer: a standalone all_reduce of one float over the same world size, same launcher, no model code. If this hangs, the training script is exonerated.
6. Bisect the world: run single-node all-GPU first, then two nodes, then full scale. The smallest failing configuration is the experiment target.
7. Set NCCL_ASYNC_ERROR_HANDLING=1 and a short TORCH_NCCL_BLOCKING_WAIT/timeout so future hangs fail fast with a rank-attributed error instead of blocking forever.

Controlled experiment: hold model, launcher, and node set fixed; vary exactly one factor per arm — (a) NCCL_IB_DISABLE=0 vs 1, (b) NCCL_SOCKET_IFNAME correct NIC vs default, (c) world size 8 (single node) vs 16 (two nodes). Three repeats per arm, fixed 120 s timeout, record pass/hang and time-to-first-collective. A single arm flipping the outcome identifies the causal factor.

Confounders: firewall or security-group rules blocking the ephemeral port range; docker networking without host network or without --ipc=host and a large shm; MTU mismatch on RoCE; ECMP hashing hiding a single bad link; a stale zombie process holding the port from a prior run; mixed NCCL versions across images.

Evidence required to declare root cause: per-rank NCCL INFO logs showing the divergent step, the minimal all_reduce reproducing and then passing after exactly one changed variable, and a clean full-scale run afterwards.

Rollback gates: if the fix is an env override (interface pinning, IB disable), treat it as a mitigation and not a cure — record it, and revert if step time regresses more than 10 percent versus the last known-good baseline, since NCCL_IB_DISABLE=1 in particular can silently drop you onto TCP. Abort the change and restore the previous launcher config if any rank fails the minimal all_reduce after the change."""

def pick(user, assistant):
    if "collective initialization" in user:
        return HANG, ["Overriding NCCL_IB_DISABLE=1 as a permanent fix silently degrades to TCP and can cut interconnect bandwidth by an order of magnitude",
                      "Blind timeout increases hide stragglers and turn fast failures into multi-hour wasted allocations",
                      "Restarting ranks before capturing py-spy stacks and NCCL logs destroys the only evidence"], \
               ["Per-rank env dump (RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/CUDA_VISIBLE_DEVICES)",
                "NCCL_DEBUG=INFO logs from every rank, diffed",
                "py-spy stack dumps of hung ranks",
                "nvidia-smi topo -m and ibstat/ibv_devinfo output",
                "Minimal all_reduce reproducer result at each world size"]
    return OOM, ["FP8/INT8 KV quantization changes model outputs and must pass an accuracy gate before production",
                 "Raising gpu_memory_utilization too close to 1.0 leaves no room for CUDA graphs or NCCL buffers and converts OOM into harder-to-debug crashes",
                 "Admission control caps trade tail latency and rejection rate for stability and must be announced to callers"], \
           ["OOM traceback with failing allocation size",
            "torch.cuda.memory_allocated vs memory_reserved at failure",
            "Aggregate active context length and num_requests_running at failure",
            "Server metrics: gpu_cache_usage_perc, preemption count, waiting queue depth",
            "Threshold reproduction matching the KV-size formula within 10 percent"]

rows = []
with open(CORPUS) as f:
    for i, line in enumerate(f):
        if i < START: continue
        if i >= START + N: break
        d = json.loads(line)
        msgs = d["messages"]
        u = [m for m in msgs if m["role"] == "user"][0]["content"]
        a = [m for m in msgs if m["role"] == "assistant"][0]["content"]
        ans, risks, ev = pick(u, a)
        rows.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": u,
            "source_assistant": a,
            "corrected_answer": ans,
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 3,
            },
            "risks": risks,
            "evidence_required": ev,
            "confidence": 0.78,
        })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(rows))
