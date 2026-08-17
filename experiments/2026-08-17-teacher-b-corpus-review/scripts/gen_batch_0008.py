import json

SRC = "/media/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT = "/media/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0008.jsonl"

HEAD_A = """Claim under review. "Prefill processes the prompt, is parallel across prompt tokens, and its cost is dominated by prompt processing / compute utilization" is directionally correct for inference but it does not actually answer how prefill *changes* between training and inference. Assumptions: dense decoder-only transformer, hidden size d, layers L, prompt length N, batch B, bf16 weights, single-node 8x A30 24GB class hardware unless stated. All numbers below are analytic estimates, not measurements.

What is structurally the same. In both regimes the forward pass over N tokens is one parallel pass: token positions are processed simultaneously under a causal mask, so the projections and MLP are dense GEMMs of shape roughly (B*N x d) x (d x d'). Arithmetic intensity is high (FLOPs ~ 2 * params * B * N for the dense part, plus ~ 2 * L * B * N^2 * d for attention scores+values), so both are compute-bound rather than memory-bandwidth-bound once B*N is a few thousand tokens.

What actually changes.
1. Activation lifetime. Training must retain (or recompute) per-layer activations for backward; inference discards them immediately. Training memory therefore scales ~ O(L * B * N * d) for stashed activations, while inference prefill peak is dominated by the largest single-layer working set plus the KV cache it emits.
2. Output artifact. Training prefill produces a loss and gradients; inference prefill produces a KV cache of size 2 * L * N * n_kv_heads * head_dim * dtype_bytes per sequence, which is then *reused* by decode. Training throws the KV away.
3. Cost accounting. Training cost is ~3x forward (1 forward + ~2x backward) and there is no decode phase at all. Inference splits into prefill (compute-bound, sets TTFT) and decode (memory-bandwidth-bound, sets ITL); the same GEMM shapes that are efficient in prefill become skinny GEMV-like ops in decode.
4. Scheduling. Training batches are homogeneous and fixed-shape, so pipeline/tensor parallel schedules are static. Inference prefill lengths are heterogeneous and arrive online, forcing chunked prefill, continuous batching, and prefill/decode interference control (or full P/D disaggregation as in Mooncake and NVIDIA Dynamo, where prefill and decode run on separate worker pools and KV blocks are transferred over RDMA).

"""

BOUND_A = {
 1: """Boundary condition (variant 1): the "prefill is compute-bound" statement fails when N is small. At N below roughly 128-256 tokens the per-layer GEMM is too skinny to saturate tensor cores, and prefill degenerates toward the decode regime: latency becomes dominated by weight reads from HBM, i.e. ~ (model_bytes / HBM_BW) per forward. On A30 (~933 GB/s, ~165 TFLOPS bf16 dense) the analytic crossover is where 2*params*N FLOPs / 165e12 exceeds params*2 bytes / 933e9, i.e. N of order 100-200. Falsifiable prediction: TTFT should be nearly flat as N grows from 32 to 128 and only then start growing linearly.""",
 2: """Boundary condition (variant 2): the linear-in-N cost model fails at long context. Attention contributes ~2 * L * B * N^2 * d FLOPs and, without FlashAttention-style tiling, an O(N^2) score materialization. Beyond roughly N = 8k-16k the quadratic term overtakes the dense term for typical d, so TTFT stops being linear in N. Falsifiable prediction: a log-log plot of TTFT vs N has slope ~1.0 in the short-context regime and trends toward ~2.0 past the crossover; if the observed slope stays at 1.0 out to 32k, the attention kernel is either tiled and bandwidth-limited or the measurement is saturating elsewhere.""",
 3: """Boundary condition (variant 3): the training/inference symmetry fails once activation checkpointing is enabled. With full recompute, training's forward is executed twice, so training-side "prefill" FLOPs rise ~33% while activation memory drops from O(L*B*N*d) to O(sqrt(L)*B*N*d) or to one layer's worth. Falsifiable prediction: enabling full recompute should cut peak training memory by a large factor while step time rises by roughly 25-35%; if step time rises far more, the bottleneck is recompute of attention at long N, not the dense layers.""",
 4: """Boundary condition (variant 4): the "prefill is a clean parallel phase" model fails under chunked prefill with concurrent decode. When a scheduler interleaves prefill chunks with running decode requests on the same GPU, prefill steals SM time and inflates ITL for decode traffic; conversely capping chunk size inflates TTFT. There is no setting that optimizes both on one worker. Falsifiable prediction: sweeping max_num_batched_tokens should trace a monotone TTFT-vs-p99-ITL frontier; if both improve together, the GPU was not saturated and the experiment was run below the knee.""",
 5: """Boundary condition (variant 5): the single-GPU cost model fails at multi-node scale. With tensor parallelism, every prefill layer inserts an all-reduce of ~2 * B * N * d bytes; that traffic is proportional to N, so on a slow fabric prefill becomes communication-bound even though it is compute-bound on one device. Falsifiable prediction: at fixed model and N, moving from NVLink-class intra-node links to RoCE/InfiniBand-crossing TP should degrade prefill throughput measurably, and NCCL all-reduce busbw measured by nccl-tests should account for the gap; if it does not, the regression is in kernel launch or scheduling, not the fabric."""
}

HEAD_B = """Claim under review. The source answer only restates the textbook description ("prefill is parallel across prompt tokens and compute-dominated") and does not do what was asked: name a misleading intuition and correct it. Assumptions: dense decoder-only transformer, prompt length N, hidden size d, layers L, bf16, A30-class GPUs; all quantitative statements are analytic estimates unless a measurement is named.

"""

MIS_B = {
 1: """Misleading intuition (variant 1): "Prefill is parallel, so doubling the prompt length costs about the same wall-clock time."
Correction and mechanism. Parallel across tokens means the work is issued in one pass, not that the work is free. Per layer the dense GEMMs are (N x d) x (d x d'), so dense FLOPs scale linearly in N, and attention adds a term scaling as N^2. Once the GPU is saturated, wall-clock TTFT grows at least linearly with N. The intuition is only true in the tiny-N regime where the GPU is underutilized.
Boundary condition: the "free" region ends at roughly N = 128-256 tokens on A30 (~933 GB/s HBM, ~165 TFLOPS bf16), where compute time first exceeds the fixed weight-read time. Falsifiable prediction: TTFT(N=64) ~= TTFT(N=128), but TTFT(N=4096) ~= 32x TTFT(N=128) plus a superlinear attention term.""",
 2: """Misleading intuition (variant 2): "Prefill and decode are both just forward passes, so one throughput number describes the server."
Correction and mechanism. They sit in different roofline regimes. Prefill has arithmetic intensity ~ O(N) and is compute-bound; decode processes one token per sequence per step, so its intensity is ~ O(batch) and it is HBM-bandwidth-bound, with step time ~ (model_bytes + KV_bytes_read) / HBM_BW. A single tokens/s figure silently mixes them and will move purely by changing the input/output length ratio.
Boundary condition: the two regimes only converge when decode batch size grows large enough to make the weight read amortized, roughly batch in the hundreds for a 9B-class model. Falsifiable prediction: report TTFT and ITL separately; if decode ITL is nearly flat as batch grows from 1 to 64, the server is bandwidth-bound as predicted, and adding FLOPs (a faster GPU at equal bandwidth) will not help.""",
 3: """Misleading intuition (variant 3): "Prefill is compute-bound, therefore prefill has no memory problem."
Correction and mechanism. Prefill is where the KV cache is *created*. Its size is 2 * L * N * n_kv_heads * head_dim * dtype_bytes per sequence; for a 9B-class model with GQA this is on the order of a few hundred KB per token, so a single 32k-token prompt can consume several GB before a single output token is produced. On 24GB A30s that is the dominant OOM source, not the transient activations.
Boundary condition: the danger zone is long-prompt, high-concurrency traffic; short prompts with long outputs never hit it. Falsifiable prediction: OOM/preemption events should correlate with sum of admitted prompt tokens, not with request count; if they correlate with request count instead, the leak is in per-request overhead or block-allocator fragmentation, not KV sizing.""",
 4: """Misleading intuition (variant 4): "Because prefill is compute-bound, adding GPUs via tensor parallelism scales prefill nearly linearly."
Correction and mechanism. TP inserts two all-reduces per transformer layer, each moving ~2 * B * N * d bytes. That collective traffic grows with N, exactly the axis you were trying to speed up, so speedup is bounded by Amdahl's law with a communication term that is not constant. On NVLink-class intra-node links the term is small; across RoCE/InfiniBand it frequently dominates, and GPUDirect RDMA (GDR) versus a staged host-memory path can change effective busbw by a large factor.
Boundary condition: TP scaling holds only while collective time stays a small fraction of layer compute; at small N or wide TP degree it does not. Falsifiable prediction: measure nccl-tests all_reduce busbw at the per-layer message size and check whether L * 2 * message_time explains the gap between ideal and observed prefill speedup. Roll back TP width if the measured gap exceeds ~20% of step time.""",
 5: """Misleading intuition (variant 5): "Prefill/decode disaggregation is strictly better, so splitting the fleet always wins."
Correction and mechanism. Disaggregation (the Mooncake and NVIDIA Dynamo style architecture) removes prefill/decode interference and lets each pool be sized to its own bottleneck, but it introduces a KV cache transfer of the full 2 * L * N * n_kv_heads * head_dim * dtype_bytes per request across the fabric between the prefill worker and the decode worker. That transfer is on the critical path of TTFT.
Boundary condition: the tradeoff flips when transfer time exceeds the interference it removed, i.e. for short prompts, weak fabrics, or when GDR is not actually engaged and the path silently falls back to host bounce buffers. Falsifiable prediction: measure end-to-end TTFT and p99 ITL for colocated versus disaggregated at matched load, and separately measure the KV transfer time; disaggregation should win only where measured interference cost exceeds measured transfer cost. Rollback gate: revert to colocated if p99 TTFT regresses more than 10% or if KV transfer failures produce any request-level errors."""
}

RISKS_A = [
 "Source answer never contrasts training and inference, so it does not answer the question asked",
 "Omits activation retention for backward, the main memory difference between the two regimes",
 "Omits the KV cache that inference prefill emits and training discards",
 "'often compute utilization' is vague and not falsifiable",
]
RISKS_B = [
 "Source answer does not name any misleading intuition, so instruction coverage is near zero",
 "Restates a generic definition that is only valid in the large-N, saturated-GPU regime",
 "No boundary condition and no measurable prediction, so the claim cannot be falsified",
]
EVID_A = [
 "Profiler trace separating dense GEMM time from attention time during prefill",
 "Peak allocator memory with and without activation checkpointing at matched batch/sequence",
 "Engine-reported KV cache bytes per sequence and per token",
 "TTFT vs prompt-length sweep with fixed output length",
]
EVID_B = [
 "TTFT vs prompt-length sweep (N = 64, 128, 512, 4096, 32768) at fixed output length",
 "Separate TTFT and per-token ITL percentiles rather than a single tokens/s number",
 "DCGM SM occupancy and HBM bandwidth utilization during prefill and decode steps",
 "nccl-tests all_reduce busbw at the per-layer message size when tensor parallelism is in play",
]

recs = []
lines = open(SRC).read().splitlines()
for ln in lines[70:80]:
    d = json.loads(ln)
    user = [m["content"] for m in d["messages"] if m["role"] == "user"][0]
    asst = [m["content"] for m in d["messages"] if m["role"] == "assistant"][0]
    v = int(user.split("Case variant ")[1].split(":")[0])
    if user.startswith("Explain how prefill changes"):
        ans = HEAD_A + BOUND_A[v]
        risks, evid = RISKS_A, EVID_A
        qd = {"technical_correctness": 2, "instruction_coverage": 1, "operational_safety": 3}
        conf = 0.86
    else:
        ans = HEAD_B + MIS_B[v]
        risks, evid = RISKS_B, EVID_B
        qd = {"technical_correctness": 2, "instruction_coverage": 1, "operational_safety": 3}
        conf = 0.87
    recs.append({
        "source_id": d["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": user,
        "source_assistant": asst,
        "corrected_answer": ans,
        "quality_dimensions": qd,
        "risks": risks,
        "evidence_required": evid,
        "confidence": conf,
    })

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(recs), OUT)
