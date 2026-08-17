import json

CORPUS = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0016.jsonl"
START, END = 151, 160  # 1-indexed inclusive

FRAME = ("Assumption frame: single node, 8x NVIDIA A30 24 GB (HBM2, ~933 GB/s theoretical peak, "
         "PCIe Gen4 x16 host links, no NVLink bridge assumed), dense ~9B decoder in bf16 (~18 GB weights), "
         "paged KV cache, vLLM-class iteration-level scheduler. All numbers are ESTIMATES from a "
         "roofline/queueing model unless explicitly labelled MEASURED.\n\n")

MECH = """Concrete mechanism. Continuous batching moves the scheduling unit from "one request" to "one decode iteration". After every forward pass the engine (a) retires sequences that emitted EOS or hit max_tokens and returns their paged KV blocks to the allocator, (b) re-runs admission against the freed block pool, and (c) rebuilds the block table / attention metadata for the new running set before the next step. Requests therefore join and leave mid-flight instead of being pinned to a batch that lives until its slowest member finishes."""

BOUND = """Boundary condition. The benefit is a function of output-length dispersion and of which resource is binding. Static batching wastes decode slot-steps in proportion to 1 - mean(L)/max(L); when every request emits nearly the same number of tokens that waste is ~0 and continuous batching only adds per-iteration scheduling and CUDA-graph-reshape overhead. Independently, once KV utilisation sits at the admission high-water mark the binding constraint is bytes, not slots: freeing a finished sequence admits a new one only if the newcomer's projected KV footprint fits, so throughput stops tracking slot availability."""

MEM = """Memory arithmetic to state explicitly. kv_bytes_per_token = 2 (K and V) * n_layers * n_kv_heads * head_dim * dtype_bytes, taken from the served checkpoint config, not assumed. Decode step time is approximately (weights_bytes + sum_i KV_bytes_i) / BW_eff, with BW_eff ~= 0.6-0.75 * 933 GB/s on A30. At 18 GB of weights and running=1 that is ~26-32 ms/step as an ESTIMATE; anything materially slower at running=1 is framework overhead, not bandwidth."""

A = {}

A["corpus-00169"] = FRAME + """How continuous batching interacts with latency, throughput and memory, variant 4.

""" + MECH + """

Interaction with throughput. Aggregate token throughput rises because padded slot-steps disappear; the ceiling is set by HBM bandwidth, not by the scheduler. In the dispersion-limited regime the gain is roughly max(L)/mean(L); in the bandwidth-limited regime adding sequences raises tokens/s sublinearly because each co-resident sequence adds its KV read to every step.

Interaction with latency. Throughput is bought with tail latency. TTFT worsens when a newly admitted long prompt runs an unchunked prefill that decoding sequences must wait behind. ITL worsens for every already-running sequence each time admission grows the running set, because per-step bytes grow. These are distinct signals and must be reported as separate percentile series; a single "latency" number hides the trade.

Interaction with memory. """ + MEM + """ Paged allocation caps internal fragmentation at under one page (commonly 16 tokens) per sequence, versus contiguous max_seq_len reservation which wastes ~1 - mean_len/max_seq_len.

""" + BOUND + """

Falsifiable hypotheses. H1: if the win is dispersion-driven, forcing all requests to identical output length shrinks the throughput gap over static batching to <10%. H2: if the binding resource is KV bytes, raising gpu_memory_utilization increases the steady-state running-sequence count; if running count does not move, H2 is false.

Evidence required before any claim. MEASURED output-length distribution (p50/p99) from production logs; engine counters for running vs waiting sequences, KV utilisation %, preemption/recompute count; per-step time at several running-set sizes; paired A/B on the same replayed trace.

Rollback gate. Revert the scheduling change if the preemption counter leaves zero at steady state, or if p99 TTFT regresses >10% at target QPS over a full peak window. Change admission policy and memory-utilisation settings in separate deploys, otherwise a regression cannot be attributed."""

A["corpus-00170"] = FRAME + """How continuous batching interacts with latency, throughput and memory, variant 5.

""" + MECH + """

The three-way coupling, stated as one budget. Every admitted sequence simultaneously (1) consumes KV blocks, (2) adds bytes to each decode step, and (3) contributes tokens/s. So throughput and per-sequence latency are not independent knobs: at fixed bandwidth, tokens/s_total and ITL_per_sequence trade off along the same curve, and KV capacity sets where that curve terminates.

Quantified illustration (ESTIMATE). With 18 GB weights and BW_eff ~= 650 GB/s, running=1 costs ~28 ms/step. If each sequence's live KV is ~200 MB, running=32 adds ~6.4 GB/step, giving ~38 ms/step: ~36% worse ITL for ~32x the aggregate tokens/s. That is the whole trade in one line, and it must be recomputed with the real kv_bytes_per_token.

""" + MEM + """

""" + BOUND + """

Failure mode to watch. Preemption thrash: admission cannot see a sequence's future KV growth, so under sustained load the allocator exhausts blocks mid-flight and victims are swapped (KV_bytes / ~25 GB/s realised PCIe Gen4 x16, each way) or dropped and re-prefilled. Goodput collapses while GPU utilisation still reads high, which is why utilisation is not a health signal here.

Falsifiable hypotheses. H1: if ITL growth is bandwidth-driven, measured step time is affine in sum_i KV_bytes_i with slope ~1/BW_eff; a non-affine curve falsifies it. H2: if TTFT spikes are prefill interference, enabling chunked prefill reduces p99 TTFT without changing steady-state tokens/s.

Evidence required. Per-step time paired with concurrent KV utilisation sampled at >=1 Hz; separate TTFT and ITL percentile series; preemption counter; checkpoint-derived kv_bytes_per_token.

Rollback gate. Hold preemption at zero and p99 ITL within SLO for a full peak window before keeping the change; otherwise roll back to the previous admission configuration."""

A["corpus-00171"] = FRAME + """Measurement plan for validating whether continuous batching helps this serving workload, variant 1.

""" + MECH + """

Step 0 - state the hypothesis in falsifiable form. H: at our production offered load and output-length distribution, iteration-level scheduling raises sustained tokens/s by >=20% without regressing p99 TTFT by more than 10%. If the measured gain is under the noise band, the change is not justified.

Step 1 - characterise the workload before touching the engine. From production logs extract input-length and output-length distributions (p50/p90/p99), arrival-rate profile over a peak window, and prefix-cache hit rate. The single most predictive statistic is max_len/mean_len dispersion; without it the experiment cannot be interpreted.

Step 2 - fix the comparison. Same checkpoint, same dtype, same max_model_len, same tensor-parallel degree, same GPU clocks (lock clocks and record power cap; A30 clock drift alone can move results several percent). Only the scheduling/batching mode differs. Replay the identical recorded trace against both arms, closed-loop QPS ladder, not a synthetic uniform-length benchmark.

Step 3 - metrics to record per arm. Sustained tokens/s (output tokens only); TTFT p50/p99; ITL p50/p99; goodput under the SLO, i.e. fraction of requests meeting both TTFT and ITL targets; running vs waiting sequence counts; KV utilisation %; preemption/recompute counter; per-step time. Report goodput, not raw throughput, as the decision metric.

Step 4 - statistics. Three independent runs per load point, discard the first 60 s warm-up, report medians with min/max. Declare a win only if the intervals do not overlap.

""" + BOUND + """

Evidence required. Replayed production trace; locked clocks and recorded power/thermal state; engine counters as above; kv_bytes_per_token derived from the checkpoint config.

Rollback gate. Promote only if goodput improves at target QPS, preemption stays at zero, and p99 TTFT regression is <10%; otherwise revert. Contamination caution: never validate on prompts that also appear in a tuning or evaluation set."""

A["corpus-00173"] = FRAME + """Measurement plan for validating whether continuous batching helps this serving workload, variant 3.

""" + MECH + """

Design: a load ladder with an attribution arm. Three arms are needed because two independent effects are usually conflated. Arm A: static request-level batching with contiguous KV. Arm B: static batching with paged KV. Arm C: continuous batching with paged KV. A-to-B isolates the memory-fragmentation win; B-to-C isolates the scheduler win. Running only A vs C produces a number you cannot attribute.

Load ladder. Sweep offered QPS from 25% to 150% of expected peak in ~6 steps, closed loop, replaying a recorded production trace. At each point hold 5 minutes after a 60 s warm-up.

Primary metric. Goodput at SLO (requests meeting TTFT and ITL targets per second), plus the saturation QPS at which p99 ITL crosses the SLO. Raw tokens/s is a secondary metric because it keeps rising while user experience degrades.

Instrumentation to sample at >=1 Hz. running/waiting counts, KV utilisation %, preemption and recompute counters, per-iteration step time, and DCGM memory-controller utilisation. Pair every latency sample with the concurrent KV utilisation, otherwise regressions cannot be explained after the fact.

Falsifiable hypotheses. H1: most of the A-to-C gain reproduces in A-to-B, i.e. the win is paged memory rather than scheduling. H2: the C-arm advantage scales with the trace's max_len/mean_len; replaying a length-homogenised trace collapses it to <10%.

""" + BOUND + """

""" + MEM + """

Evidence required. Recorded production trace (not synthetic); locked GPU clocks; per-arm counter series; checkpoint-derived kv_bytes_per_token; three repeats per load point.

Rollback gate. Adopt arm C only if it strictly dominates on goodput at and below target QPS with preemption at zero; if it wins on tokens/s but loses on p99 ITL, do not ship. Do not change more than one variable per deploy."""

A["corpus-00174"] = FRAME + """Measurement plan for validating whether continuous batching helps this serving workload, variant 4.

""" + MECH + """

What is actually being tested. Not "is continuous batching faster" in general, which is settled, but "does our workload sit in the regime where it helps, at our SLO". The plan must be able to return "no".

Pre-registration. Before running anything, write down: target QPS, TTFT and ITL SLOs, the minimum goodput improvement worth the operational risk, and the abort conditions. Pre-registration prevents post-hoc metric shopping when the first result is ambiguous.

Procedure.
1. Baseline capture: 24 h of production request logs; compute input/output length distributions and the arrival profile.
2. Isolation: dedicated node, locked SM and memory clocks, fixed power cap, no co-tenant processes; record nvidia-smi topo -m and driver/framework versions in the run record.
3. Trace replay at a QPS ladder (25/50/75/100/125/150% of peak), closed loop, 3 repeats, 60 s warm-up discarded, 5 min steady state.
4. Arms: current production config vs continuous-batching config, one variable changed.
5. Collect per-request TTFT/ITL and engine counters; compute goodput at SLO.

Decision rule. Ship if goodput at target QPS improves beyond the pre-registered threshold with non-overlapping run-to-run ranges, preemption stays at zero, and p99 TTFT regression is <10%. Otherwise the hypothesis is rejected for this workload.

Failure modes that invalidate the measurement. Open-loop load generators that let queueing hide behind client-side backlog; synthetic uniform-length prompts (they erase the dispersion the change exploits); unlocked clocks; warm prefix cache in one arm only; and comparing runs taken on different days with different background load.

""" + BOUND + """

Evidence required. The counter series above, the recorded trace, environment capture, and the pre-registration document. Rollback gate: revert on any preemption at steady state or SLO regression, and never enable admission-aggressiveness and memory-utilisation changes in the same deploy."""

A["corpus-00176"] = FRAME + """Assumptions that must be stated before making a performance claim about continuous batching, variant 1.

""" + MECH + """

Assumptions about the workload. (1) Output-length distribution, specifically mean and max: the entire dispersion argument is quantitative and a claim without max_len/mean_len is unfalsifiable. (2) Input-length distribution and prefix-cache hit rate, since prefill cost and cache reuse dominate TTFT. (3) Arrival process and whether the measurement was open or closed loop. (4) Whether the trace is production-replayed or synthetic.

Assumptions about the model and memory. kv_bytes_per_token, computed as 2 * n_layers * n_kv_heads * head_dim * dtype_bytes from the served checkpoint, plus weight bytes and dtype. Without these, KV utilisation and concurrency limits cannot be derived, and every throughput claim silently depends on them. """ + MEM + """

Assumptions about hardware and configuration. GPU model and HBM bandwidth, whether clocks were locked, power cap, tensor-parallel degree, interconnect topology (nvidia-smi topo -m; on A30 without NVLink, TP traffic crosses PCIe), max_model_len, gpu_memory_utilization, chunked-prefill on/off, and framework/driver versions.

Assumptions about the metric. Which throughput (output tokens/s vs total tokens/s), which latency (TTFT vs ITL vs end-to-end), which percentile, whether goodput-at-SLO or raw throughput, warm-up discarded, and how many repeats.

""" + BOUND + """

Falsifiable hypothesis implied by any such claim. "Gain G holds because dispersion is D": replay a length-homogenised trace and G should collapse below 10%. If it does not, the stated mechanism is wrong even if the number is real.

Evidence required. The full environment capture, checkpoint-derived KV arithmetic, counter series (running/waiting, KV utilisation, preemption), and repeated runs. Rollback gate for anything shipped on the basis of the claim: preemption at zero and p99 within SLO for a full peak window, else revert."""

A["corpus-00177"] = FRAME + """Assumptions that must be stated before making a performance claim about continuous batching, variant 2.

""" + MECH + """

Rule of thumb: a performance claim is only meaningful if a skeptical reader can compute a rough prediction and disagree with you. That requires the following to be on the page.

1. Regime. Is the deployment bandwidth-bound (decode-dominant, large weights, small batch), compute-bound (prefill-dominant, long prompts), or capacity-bound (KV utilisation pinned)? Continuous batching helps mainly in the first regime and is neutral-to-harmful in the third once preemption starts.
2. Baseline identity. What exactly is "before"? Static request-level batching with contiguous KV and static batching with paged KV are very different baselines, and a gain measured against the former is mostly a memory-fragmentation result, not a scheduler result.
3. Length dispersion. mean and p99 output length; the padded-slot waste is ~1 - mean(L)/max(L).
4. KV arithmetic. """ + MEM + """
5. Concurrency actually achieved. Running-sequence count at steady state, not the configured maximum.
6. Preemption/recompute count. A throughput number measured while preemption is non-zero is not reproducible.
7. Environment. GPU, clocks locked or not, power cap, TP degree, topology, chunked prefill, framework version.
8. Statistical treatment. Repeats, warm-up handling, and dispersion across runs.

""" + BOUND + """

Falsifiable hypotheses. H1: the claim is scheduler-driven, so it survives when the baseline also uses paged KV. H2: the claim is bandwidth-limited, so measured step time is affine in total live KV bytes with slope ~1/BW_eff.

Evidence required. Environment capture, checkpoint-derived kv_bytes_per_token, paired trace replay, per-step time vs KV utilisation series, preemption counter, >=3 repeats. Rollback gate: any shipped change reverts if preemption leaves zero or p99 TTFT/ITL regresses beyond SLO during a full peak window. Contamination caution: benchmark prompts must be isolated from tuning data or the number measures workload memorisation, not scheduling."""

A["corpus-00178"] = FRAME + """Assumptions that must be stated before making a performance claim about continuous batching, variant 3.

""" + MECH + """

Framed as the assumptions that, if wrong, flip the conclusion.

A1. "Output lengths are dispersed." If false (fixed-length scoring, single-token classification, forced decoding), padded-slot waste is ~0 and the claimed gain evaporates; residual per-iteration scheduling and CUDA-graph reshape overhead can make continuous batching slower.
A2. "Slots, not KV bytes, were the binding constraint in the baseline." If false, the baseline was already capacity-bound and the scheduler cannot admit more work; the observed gain, if any, came from paged allocation.
A3. "The system was at steady state without preemption." If false, the measurement includes swap (KV_bytes / ~25 GB/s realised PCIe Gen4 x16 each way) or full prefix recompute, and is not reproducible.
A4. "Throughput is the objective." If the SLO is tail latency, higher tokens/s with worse p99 ITL is a regression, and the claim is measuring the wrong thing.
A5. "Load generation was closed-loop against a production-shaped trace." If false, client-side queueing or synthetic uniform lengths dominate the result.
A6. "Hardware state was pinned." Unlocked clocks, thermal drift, or a co-tenant process on the A30 can produce several percent of spurious difference.
A7. "The KV arithmetic uses the real checkpoint." """ + MEM + """

""" + BOUND + """

Falsifiable hypotheses. H1: homogenising output lengths collapses the gain to <10% (tests A1). H2: adding paged KV to the static baseline recovers most of the gain (tests A2).

Evidence required. Counter series for running/waiting and KV utilisation, preemption counter identically zero, locked-clock environment capture, checkpoint-derived kv_bytes_per_token, >=3 repeats with warm-up discarded, and separate TTFT/ITL percentile series.

Rollback gate. Revert on any non-zero steady-state preemption or p99 SLO breach over a full peak window; change one variable per deploy so a regression is attributable."""

A["corpus-00179"] = FRAME + """Assumptions that must be stated before making a performance claim about continuous batching, variant 4.

""" + MECH + """

Minimum disclosure set, in the order a reviewer will ask for it.

Model and memory. Parameter count, dtype and weight bytes (here ~9B bf16, ~18 GB); n_layers, n_kv_heads, head_dim and the resulting kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes; whether KV is quantised; max_model_len; gpu_memory_utilization and the derived number of KV blocks.

Hardware. GPU SKU and HBM bandwidth (A30, ~933 GB/s theoretical; assume BW_eff ~0.6-0.75 of that), PCIe generation and topology from nvidia-smi topo -m, presence or absence of NVLink, clock-lock state, power cap, and whether the node was exclusive.

Serving configuration. TP/PP degree, chunked prefill on/off, scheduler admission limits (max_num_seqs, max_num_batched_tokens), speculative decoding on/off, CUDA graph capture on/off, framework and driver versions.

Workload. Trace provenance (production replay vs synthetic), input/output length distributions with p50 and p99, prefix-cache hit rate, arrival process, and open vs closed loop.

Measurement. Metric definition (output tokens/s; TTFT vs ITL; percentile), goodput-at-SLO, warm-up discarded, repeat count, run-to-run spread.

""" + BOUND + """

Falsifiable hypotheses. H1: step time is affine in total live KV bytes with slope ~1/BW_eff; a flat curve means the system is not bandwidth-bound and the throughput story is wrong. H2: the gain tracks max_len/mean_len across traces.

Evidence required. All of the disclosure set above as captured artefacts, plus per-step time paired with KV utilisation and a preemption counter pinned at zero. Rollback gate: revert any shipped configuration if preemption becomes non-zero at steady state or p99 latency breaches SLO during a full peak window. Never present a number produced on a benchmark whose prompts overlap tuning data."""

A["corpus-00180"] = FRAME + """Assumptions that must be stated before making a performance claim about continuous batching, variant 5.

""" + MECH + """

The claim template. "Compared to BASELINE, on WORKLOAD, at LOAD, with CONFIG, on HARDWARE, metric METRIC improved by X% (median of N runs, spread S), with preemption at zero and p99 TTFT/ITL within SLO." Every capitalised slot is an assumption that must be filled or the claim is not checkable.

Why each slot matters, briefly. BASELINE decides whether you are measuring scheduling or paged memory. WORKLOAD's length dispersion sets the theoretical ceiling (~max_len/mean_len). LOAD decides the regime: below the crossover, decode is dominated by the ~18 GB weight read and batching of any flavour changes little. CONFIG (chunked prefill, max_num_seqs, gpu_memory_utilization, CUDA graphs) can move results more than the batching mode itself. HARDWARE with locked clocks and known topology makes the roofline computable. METRIC must be goodput-at-SLO if the service has latency targets.

""" + MEM + """

""" + BOUND + """

Falsifiable hypotheses. H1: the reported gain reproduces on a second, independently recorded production trace with similar dispersion. H2: forcing uniform output lengths collapses the gain to <10%. H3: measured step time at running=1 matches the roofline estimate (~26-32 ms here) within noise; a large constant excess indicates framework overhead rather than a batching effect.

Evidence required. Filled claim template with artefacts, environment capture, checkpoint-derived kv_bytes_per_token, per-step time vs KV utilisation series at >=1 Hz, preemption counter, >=3 repeats with warm-up discarded, and separate TTFT/ITL percentile series.

Rollback gate and safety. Ship only if goodput improves at and below target QPS with preemption at zero and no p99 regression beyond SLO across a full peak window; revert otherwise. Change one variable per deploy. Do not quote a number obtained with evaluation prompts that overlap tuning data, and do not describe an ESTIMATE from roofline arithmetic as MEASURED."""

DEC = {
 "corpus-00169": ("rewrite", 3, 1, 2, 0.62),
 "corpus-00170": ("rewrite", 3, 1, 2, 0.62),
 "corpus-00171": ("rewrite", 3, 1, 2, 0.6),
 "corpus-00173": ("rewrite", 3, 1, 2, 0.6),
 "corpus-00174": ("rewrite", 3, 1, 2, 0.6),
 "corpus-00176": ("rewrite", 3, 1, 2, 0.6),
 "corpus-00177": ("rewrite", 3, 1, 2, 0.6),
 "corpus-00178": ("rewrite", 3, 1, 2, 0.61),
 "corpus-00179": ("rewrite", 3, 1, 2, 0.61),
 "corpus-00180": ("rewrite", 3, 1, 2, 0.61),
}

GEN_RISKS = [
 "Source answer is a single generic sentence: it does not answer the asked question type (measurement plan / assumption list) and omits the required boundary condition, so it would teach topic-shaped non-answers.",
 "No KV arithmetic, no regime distinction (bandwidth- vs capacity-bound), so a model trained on it may assert throughput gains that do not hold once KV utilisation is pinned.",
 "No operational guardrails: omits preemption/recompute thrash, tail-latency regression, and the requirement to change one variable per deploy.",
]
MP_RISKS = [
 "Source answer gives no measurement procedure at all for a question explicitly asking for one; instruction coverage is effectively zero.",
 "Absent load-generation guidance invites open-loop or synthetic uniform-length benchmarks, which erase the dispersion effect being tested and produce unreproducible numbers.",
 "No baseline identity, warm-up, repeat-count or clock-locking discipline, so results could be reported without attribution or error bars.",
]
AS_RISKS = [
 "Source answer lists no assumptions for a question that asks only for assumptions; it is off-task.",
 "Encourages unfalsifiable performance claims stated without model/hardware/workload disclosure or metric definition.",
 "Omits the contamination caution that benchmark prompts must be isolated from tuning data, and the ESTIMATE-vs-MEASURED distinction.",
]

RISKS = {
 "corpus-00169": GEN_RISKS, "corpus-00170": GEN_RISKS,
 "corpus-00171": MP_RISKS, "corpus-00173": MP_RISKS, "corpus-00174": MP_RISKS,
 "corpus-00176": AS_RISKS, "corpus-00177": AS_RISKS, "corpus-00178": AS_RISKS,
 "corpus-00179": AS_RISKS, "corpus-00180": AS_RISKS,
}

GEN_EV = [
 "MEASURED output-length distribution (p50/p99) from production logs, giving max_len/mean_len dispersion",
 "Engine counters sampled at >=1 Hz: running/waiting sequences, KV utilisation %, preemption/recompute count, per-iteration step time",
 "kv_bytes_per_token derived from the served checkpoint config, plus locked-clock environment capture (nvidia-smi topo -m, power cap, framework/driver versions)",
]
MP_EV = [
 "Recorded production trace replayed closed-loop over a QPS ladder, >=3 repeats with 60 s warm-up discarded",
 "Per-arm goodput-at-SLO with separate TTFT and ITL percentile series, plus preemption counter held at zero",
 "Attribution arm with paged KV under static batching, to separate memory-fragmentation gains from scheduler gains",
]
AS_EV = [
 "Full disclosure set: model/dtype/weight bytes, n_layers/n_kv_heads/head_dim, max_model_len, gpu_memory_utilization, TP degree, chunked-prefill state",
 "Hardware capture: GPU SKU and HBM bandwidth, PCIe topology from nvidia-smi topo -m, clock-lock and power-cap state, exclusive-node confirmation",
 "Measurement discipline: metric definition, warm-up handling, repeat count and run-to-run spread; confirmation that evaluation prompts are isolated from tuning data",
]

EV = {
 "corpus-00169": GEN_EV, "corpus-00170": GEN_EV,
 "corpus-00171": MP_EV, "corpus-00173": MP_EV, "corpus-00174": MP_EV,
 "corpus-00176": AS_EV, "corpus-00177": AS_EV, "corpus-00178": AS_EV,
 "corpus-00179": AS_EV, "corpus-00180": AS_EV,
}

rows = []
with open(CORPUS) as f:
    for i, line in enumerate(f, 1):
        if i < START: continue
        if i > END: break
        d = json.loads(line)
        sid = d["id"]
        u = [m for m in d["messages"] if m["role"] == "user"][0]["content"]
        a = [m for m in d["messages"] if m["role"] == "assistant"][0]["content"]
        dec, tc, ic, os_, conf = DEC[sid]
        rows.append({
            "source_id": sid,
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": dec,
            "source_user": u,
            "source_assistant": a,
            "corrected_answer": A[sid],
            "quality_dimensions": {"technical_correctness": tc, "instruction_coverage": ic, "operational_safety": os_},
            "risks": RISKS[sid],
            "evidence_required": EV[sid],
            "confidence": conf,
        })

assert len(rows) == 10, len(rows)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(rows))
