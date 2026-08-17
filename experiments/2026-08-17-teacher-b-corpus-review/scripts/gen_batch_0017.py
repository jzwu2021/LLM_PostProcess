import json

FRAME = ("Assumption frame: single node, 8x NVIDIA A30 24 GB (HBM2, ~933 GB/s theoretical peak, PCIe Gen4 x16 host "
         "links, no NVLink bridge assumed), dense ~9B decoder in bf16 (~18 GB weights), paged KV cache, vLLM-class "
         "iteration-level scheduler. Every number below is an ESTIMATE from a roofline/queueing model unless "
         "explicitly labelled MEASURED.\n\n")

MECH_TI = ("Concrete mechanism. Training and inference do not share a scheduling unit. Training batches are static: a "
           "global batch is fixed before the step, all sequences are padded or packed to a common length, forward and "
           "backward run to completion, and the optimizer step is a hard synchronisation barrier across all data-parallel "
           "ranks. Inference under continuous batching schedules at iteration granularity: after each decode forward the "
           "engine retires finished sequences, returns their KV blocks to the paged allocator, re-runs admission against "
           "the freed pool, and rebuilds block tables and attention metadata before the next step. Sequence membership is "
           "therefore mutable in inference and immutable in training.\n\n")

BOUND_TI = ("Boundary condition. Continuous batching does not transfer to training because the backward pass requires the "
            "activation graph of exactly the sequences in the forward pass; you cannot admit a new sequence mid-step "
            "without either recomputing or corrupting gradient accumulation, and dynamic membership would make the "
            "effective batch size (hence the gradient estimator and the LR schedule) non-deterministic and "
            "non-reproducible. The closest legitimate training analogue is sequence packing plus a block-diagonal "
            "attention mask, which removes padding waste but keeps membership fixed for the whole step.\n\n")

FALS_TI = ("Falsifiable hypotheses. H1: at fixed hardware, switching a decode-heavy workload from static batching to "
           "continuous batching raises output tokens/s by roughly max(L)/mean(L) in the length-dispersion-limited regime "
           "and by far less once per-step KV bytes saturate HBM bandwidth. H2: applying the same dynamic-membership idea "
           "to a training step changes the loss curve at fixed seed, which is exactly the observable that falsifies "
           "'continuous batching also helps training'.\n\n")

EVID_TI = ("Evidence required. Serving side: per-request TTFT and ITL percentile series (p50/p95/p99) reported separately, "
           "not a single latency number; scheduler counters for running/waiting/preempted sequences and KV block "
           "utilisation; achieved HBM bandwidth from a profiler. Training side: tokens/s and MFU with and without "
           "sequence packing, plus a fixed-seed loss-curve diff to prove the estimator was unchanged. Model side: "
           "kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes read from the served checkpoint "
           "config, never assumed.\n\n")

ROLL_TI = ("Rollback gate. Promote a scheduler or packing change only if p99 ITL regression stays under 10 percent at the "
           "target concurrency and no preemption/recompute storm appears in the counters; otherwise revert to the previous "
           "max_num_seqs / gpu_memory_utilization values in one step and re-measure before any further tuning.")

VAR_TI = {
 1: ("Variant 1 focus: the scheduling-unit contrast itself. The single most load-bearing difference is that inference "
     "membership changes at iteration boundaries while training membership is frozen by the optimizer barrier.\n\n"),
 2: ("Variant 2 focus: memory lifetime. In training, peak memory is dominated by activations and optimizer state and is "
     "known before the step, so it can be capacity-planned exactly. In inference under continuous batching, peak memory "
     "is dominated by KV cache whose occupancy is a function of live sequence lengths and therefore drifts at runtime; "
     "admission control, not a static batch size, is what keeps you off the OOM cliff.\n\n"),
 3: ("Variant 3 focus: synchronisation. Training has a mandatory all-reduce (NCCL) per step, so stragglers cost every "
     "rank. Inference decode steps under tensor parallelism also all-reduce, but there is no optimizer barrier, so a slow "
     "or long sequence delays only the steps it participates in and can be preempted and recomputed instead of blocking "
     "the run.\n\n"),
 4: ("Variant 4 focus: padding versus fragmentation. Training wastes compute on right-padding unless sequences are "
     "packed. Continuous batching removes padded slot-steps entirely, but replaces that waste with KV block "
     "fragmentation and preemption overhead, which is why block size and gpu_memory_utilization become the tuning "
     "knobs that batch_size used to be.\n\n"),
 5: ("Variant 5 focus: objective. Training optimises tokens/s and MFU subject to a fixed statistical batch, a single "
     "scalar goal. Inference optimises a two-sided SLO: throughput at fixed p99 TTFT and p99 ITL. Continuous batching is "
     "the mechanism that lets you trade along that curve at runtime, which is a control problem training does not have.\n\n"),
}

MECH_MI = ("Concrete mechanism to anchor the correction. The scheduler retires finished sequences at each iteration "
           "boundary, frees their paged KV blocks, re-runs admission against the freed pool, and rebuilds block tables "
           "and attention metadata before the next forward. Nothing about that mechanism makes a single token cheaper; "
           "it only removes idle slot-steps.\n\n")

BOUND_MI = ("Boundary condition. The benefit collapses in two regimes: (a) when sequence lengths are nearly uniform, "
            "because there is almost no dispersion to reclaim, and (b) when per-step KV reads already saturate HBM "
            "bandwidth, because admitting more sequences then adds bytes per step without adding parallel work.\n\n")

EVID_MI = ("Evidence required. Separate TTFT and ITL percentile series at matched concurrency; scheduler counters for "
           "running/waiting/preempted and KV block utilisation; achieved HBM bandwidth; a length-distribution histogram "
           "of the real traffic. A throughput-only benchmark cannot confirm or refute any of this.\n\n")

ROLL_MI = ("Rollback gate. If p99 ITL degrades more than 10 percent at target concurrency, or preemption/recompute counts "
           "become non-zero and rising, revert max_num_seqs and gpu_memory_utilization to the last known-good pair in a "
           "single change and re-measure before tuning again.")

VAR_MI = {
 1: ("Misleading intuition: 'continuous batching makes inference faster.' Correction: it does not reduce single-request "
     "latency at all; at fixed load it usually makes p99 inter-token latency slightly worse, because each newly admitted "
     "sequence adds its KV read to every subsequent decode step. What it improves is throughput at fixed hardware by "
     "eliminating slot-steps that a static batch would spend waiting for its slowest member. Falsifiable form: at "
     "concurrency 1 the two schedulers must be statistically indistinguishable on TTFT and ITL; if a benchmark shows "
     "continuous batching 'winning' at concurrency 1, the benchmark is measuring something else.\n\n"),
 2: ("Misleading intuition: 'continuous batching means you no longer need to think about batch size.' Correction: the "
     "knob moved, it did not disappear. max_num_seqs plus the KV block pool implied by gpu_memory_utilization now set the "
     "admission ceiling, and setting them too high does not fail loudly, it silently trades ITL and triggers preemption "
     "and recompute. Falsifiable form: sweep max_num_seqs and you will see a throughput plateau with a monotonically "
     "worsening p99 ITL past a threshold; if no such knee exists the workload is not KV-bound and the sweep is "
     "uninformative.\n\n"),
 3: ("Misleading intuition: 'continuous batching removes head-of-line blocking.' Correction: it removes batch-level "
     "head-of-line blocking, not prefill-level. A newly admitted long prompt whose prefill runs unchunked still stalls "
     "every decoding sequence for that iteration, which shows up as an ITL spike correlated with prompt length. Chunked "
     "prefill or a prefill/decode split is the mechanism that addresses that; continuous batching alone does not. "
     "Falsifiable form: correlate ITL spikes against admitted prompt token counts; a positive correlation refutes the "
     "claim.\n\n"),
 4: ("Misleading intuition: 'continuous batching also helps training.' Correction: it cannot be applied to a training "
     "step, because the backward pass needs the activation graph of exactly the sequences in the forward pass, and "
     "mutable membership would make the effective batch size and therefore the gradient estimator non-deterministic. "
     "The legitimate training analogue is sequence packing with a block-diagonal mask, which removes padding waste while "
     "keeping membership fixed. Falsifiable form: a correct packing change leaves the fixed-seed loss curve unchanged; "
     "any drift means the estimator was altered.\n\n"),
 5: ("Misleading intuition: 'higher GPU utilisation percentage proves continuous batching is working.' Correction: "
     "nvidia-smi utilisation reports whether any kernel was resident, not whether useful work was done; a decode loop "
     "spinning on memory-bound attention reads can show high utilisation while achieved FLOP/s is a small fraction of "
     "peak. The load-bearing metrics are achieved HBM bandwidth, output tokens/s, and the TTFT/ITL percentile pair. "
     "Falsifiable form: profile a run at reported 95+ percent utilisation and compare achieved bandwidth against the "
     "~933 GB/s A30 ceiling; a large gap refutes the utilisation-as-proof claim.\n\n"),
}

RECS = {}
for v in range(1, 6):
    RECS[180 + v] = (FRAME + VAR_TI[v] + MECH_TI + BOUND_TI + FALS_TI + EVID_TI + ROLL_TI)
for v in range(1, 6):
    RECS[185 + v] = (FRAME + VAR_MI[v] + MECH_MI + BOUND_MI + EVID_MI + ROLL_MI)

RISKS = ["source answer is a single sentence and omits the training-vs-inference boundary, so it is not usable as a "
         "gold target without expansion",
         "no assumption frame, no units, and no separation of measured facts from estimates",
         "silently invites transferring an inference-only scheduling idea to training, which would change the gradient "
         "estimator"]
EVREQ = ["separate TTFT and ITL percentile series (p50/p95/p99) at matched concurrency, not a single latency number",
         "scheduler counters: running/waiting/preempted sequences and KV block utilisation",
         "achieved HBM bandwidth from a profiler compared against the ~933 GB/s A30 ceiling",
         "kv_bytes_per_token derived from the served checkpoint config, not assumed",
         "fixed-seed loss-curve diff for any training-side packing change"]

src = {}
with open('research/ai-infra-expert/corpus/train.jsonl') as f:
    for i, line in enumerate(f, 1):
        if 161 <= i <= 170:
            d = json.loads(line)
            m = {x['role']: x['content'] for x in d['messages']}
            src[d['id']] = (m['user'], m['assistant'], i)

out = []
for sid in sorted(src):
    u, a, i = src[sid]
    n = int(sid.split('-')[1])
    out.append({
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": RECS[n],
        "quality_dimensions": {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3},
        "risks": RISKS,
        "evidence_required": EVREQ,
        "confidence": 0.72,
    })

with open('experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0017.jsonl', 'w') as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(out))
