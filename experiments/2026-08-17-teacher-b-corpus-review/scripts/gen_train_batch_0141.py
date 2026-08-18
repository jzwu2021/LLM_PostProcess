import json

CORPUS = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0141.jsonl"
START, N = 1400, 10

HDR = ("Assumptions (stated, not measured): a single-model LLM inference service on NVIDIA GPUs behind a paged "
       "KV-cache engine; the OOM surfaces from the CUDA caching allocator during serving, not at model load; "
       "model weights, dtype, tensor-parallel degree and engine version are held fixed for every comparison "
       "below; no vendor- or framework-specific behaviour is asserted beyond what the listed measurements "
       "would themselves establish.\n\n")

ANGLES = [
 ("Failure-envelope reconstruction before any hypothesis is ranked",
  "Hypothesis H1 (falsifiable): the OOM has a single dominant precondition that is visible in a 60-second window of pre-failure telemetry. If true, at least 4 of 5 captured failures share the same signature (same phase, comparable in-flight token total, comparable free-block count) rather than being scattered across unrelated states.",
  ["Capture, for every OOM, a fixed record: engine phase, in-flight sequence count, summed in-flight tokens, free KV blocks, allocator allocated/reserved/requested bytes, and device free bytes.",
   "Cluster at least five captured failures on that record; a diagnosis is only actionable if the cluster is tight. Report the spread, not just the mean.",
   "Controlled experiment: none yet. Deliberately spend the first cycle on observability, because ranking mitigations on one anecdotal failure is the most common way this investigation goes wrong."],
  "Confounders: sampling telemetry at 10 s resolution aliases a spike that lasts 200 ms and will make unrelated failures look identical; a restart between failures resets fragmentation state and breaks comparability.",
  "Rollback: this step deploys only logging. If instrumentation overhead raises p99 latency by more than 3%, reduce sampling rate rather than removing the record.",
  "Boundary condition: if the five failures do not cluster, no single mitigation is justified and the workload must be partitioned by tenant or request class before continuing."),

 ("Token-budget arithmetic: the residency inequality that must be checked first",
  "Hypothesis H2 (falsifiable): aggregate KV residency at the observed high-water mark exceeds the provisioned KV pool. If true, the inequality sum(tokens_in_flight) x per_token_KV_bytes > pool_bytes holds at the failure instant using measured, not assumed, token counts.",
  ["Compute per_token_KV_bytes = 2 x num_layers x num_kv_heads x head_dim x dtype_bytes, showing every term with units, and verify it against measured pool occupancy at a known in-flight token count rather than trusting the formula alone.",
   "Log summed in-flight tokens (prompt + generated so far) at every scheduler tick; the peak of this series, not the request count, is the quantity the pool must cover.",
   "Controlled experiment: drive a synthetic ramp of concurrency at fixed 4 k-token context until the first failure; the measured failure point should match the predicted point within 10% if H2 holds."],
  "Confounders: prefix sharing makes summed in-flight tokens overcount actual residency; speculative decoding and beam search allocate KV for tokens that are later discarded, so logical and physical token counts diverge.",
  "Rollback: none required (measurement only), but if the predicted and measured failure points differ by more than 25%, discard the arithmetic model instead of patching it with fudge factors.",
  "Boundary condition: if the inequality does not hold at failure, the pool is not exhausted and the cause lies outside KV accounting - transient workspace, fragmentation, or a non-KV allocation."),

 ("Non-KV memory consumers: workspace, CUDA graphs and communication buffers",
  "Hypothesis H3 (falsifiable): the memory lost to non-KV consumers (attention workspace, logits and sampling buffers, captured CUDA graphs, NCCL/communication buffers) is large enough that the KV pool was over-provisioned at startup and cannot honour its own budget. If true, measured steady-state reserved bytes exceed weights + KV pool by a margin comparable to the failing allocation.",
  ["Measure device memory immediately after warm-up with concurrency 1: attribute bytes to weights, KV pool, captured graphs, and everything else. The residual term is the one that is usually unbudgeted.",
   "Measure the logits/sampling buffer scaling: it grows with batch size x vocabulary size and is often the largest transient at high concurrency with a large vocabulary.",
   "Controlled experiment: disable CUDA-graph capture (or reduce captured batch sizes) with all other settings fixed; record freed bytes, OOM count and per-step latency."],
  "Confounders: disabling graph capture also slows decode, so an OOM improvement is entangled with a throughput regression; some frameworks lazily grow workspace, so a short warm-up under-reports the true steady state.",
  "Rollback: restore graph capture if decode throughput regresses more than 15% without a corresponding OOM reduction.",
  "Boundary condition: if the unattributed residual is small relative to the failing allocation, this lever cannot explain the failure and should be dropped rather than tuned."),

 ("Output-length uncertainty: reserving for generation that has not happened yet",
  "Hypothesis H4 (falsifiable): OOM is driven by requests whose *generated* output grows well past what admission assumed, so memory that fit at admission time no longer fits mid-generation. If true, failures correlate with decode step index of long-running requests rather than with arrival bursts.",
  ["Log, per failing window, the decode step index and remaining max_tokens of every in-flight request; a memory problem created at admission looks different from one created 2000 steps later.",
   "Compare the distribution of requested max_tokens with actually generated tokens; over-large max_tokens forces either pessimistic reservation or optimistic overcommit, and each fails differently.",
   "Controlled experiment: replay one arm with admission based on requested max_tokens (pessimistic) and one with overcommit plus preemption enabled, holding the trace fixed."],
  "Confounders: preemption/swap converts an OOM into a latency regression, so the failure simply changes shape and a naive 'OOM count' metric will look like success; stop conditions differ between arms if sampling parameters drift.",
  "Rollback: if preemption raises p99 TTFT beyond the SLO or produces recompute storms, revert to pessimistic admission within one deploy cycle.",
  "Boundary condition: if generated lengths are tightly clustered and short, output-length uncertainty is refuted as a driver regardless of how large max_tokens is set."),

 ("Preemption and swap policy as the intended relief valve",
  "Hypothesis H5 (falsifiable): the engine has a preemption path that should have absorbed this pressure and it is either disabled or thrashing. If true, preemption counters are zero at failure (disabled) or extremely high with repeated re-preemption of the same requests (thrashing).",
  ["Read preemption/recompute/swap counters per minute and correlate with the failure timestamps; a healthy relief valve shows nonzero but non-escalating counts.",
   "Measure recompute cost per preemption (tokens re-prefilled) to quantify the throughput price of using this valve at the observed rate.",
   "Controlled experiment: identical trace with preemption enabled versus disabled; record OOM count, preemption count, p99 TTFT and throughput."],
  "Confounders: swap-to-host moves the bottleneck onto PCIe bandwidth and may degrade throughput far more than the memory relief is worth; recompute-based preemption interacts with prefix caching and may be cheaper than it appears.",
  "Rollback: disable swap-based preemption if host-to-device transfer saturates PCIe or if throughput falls more than 20%.",
  "Boundary condition: preemption cannot help when a single request alone exceeds the pool - there is nothing left to preempt at that point."),

 ("Multi-tenant and noisy-neighbour effects on a shared device",
  "Hypothesis H6 (falsifiable): another process (a second model, an eval job, a monitoring agent, or MPS/MIG co-tenancy) intermittently consumes device memory, so the serving process' own accounting is correct but its free-memory assumption is not. If true, device free bytes drop before failure without a corresponding rise in the serving process' allocated bytes.",
  ["Sample per-process device memory usage (all PIDs on the device) at 1 s resolution and align with the failure timestamps.",
   "Verify exclusivity assumptions explicitly: MIG partitioning, MPS, and any scheduler that may co-locate jobs on the same device.",
   "Controlled experiment: reproduce the trace on a device with verified exclusive access; if OOM disappears with no engine-side change, the noisy-neighbour hypothesis is supported."],
  "Confounders: some monitoring tools themselves allocate device memory; a co-tenant that is idle during the reproduction run makes the exclusive-device arm falsely look like a fix.",
  "Rollback: not applicable to measurement; if isolation is enforced by policy, revert only if capacity utilisation targets are missed.",
  "Boundary condition: if the serving process' own allocated bytes account for essentially all device memory at failure, co-tenancy is refuted and internal accounting is the right place to look."),

 ("Batch-shape sensitivity: peak transient memory is not monotone in request count",
  "Hypothesis H7 (falsifiable): peak transient memory is driven by the *shape* of the batch (a few very long sequences versus many short ones) rather than by its cardinality. If true, a batch of k long sequences reaches a higher peak reserved than a batch of 4k short sequences with the same total token count.",
  ["Run a shape sweep at fixed total tokens: vary (num_sequences, sequence_length) across at least four points and record peak reserved bytes for each.",
   "Record whether attention kernels select different code paths at different sequence lengths - a kernel switch can change workspace by a large factor at one threshold.",
   "Controlled experiment: hold total tokens constant and vary only shape; any nonlinearity in the resulting curve identifies the threshold to defend with admission rules."],
  "Confounders: padding behaviour differs between kernels, so 'total tokens' may not be the invariant it appears to be; sequence-length bucketing in the scheduler can hide the threshold.",
  "Rollback: if a shape-aware admission rule reduces throughput more than 15%, revert to cardinality-based limits and accept a lower concurrency ceiling instead.",
  "Boundary condition: if the peak-reserved curve is flat across shapes at fixed total tokens, shape sensitivity is refuted and simple token-budget admission is sufficient."),

 ("Regression attribution: was this introduced by a change rather than by load?",
  "Hypothesis H8 (falsifiable): the OOM was introduced by a specific configuration, engine or model change rather than by traffic growth. If true, replaying a fixed archived trace against the previous build reproduces no OOM while the current build does.",
  ["Establish the change inventory across the window in which failures began: engine version, model revision, dtype, max_model_len, KV fraction, kernel/driver versions.",
   "Replay one archived trace against old and new builds on the same hardware; this is the only comparison that separates a code regression from a load change.",
   "Controlled experiment: bisect the change inventory only after the two-build comparison shows a difference; bisecting before that wastes cycles on a load problem."],
  "Confounders: traffic composition drifts, so a live A/B across time is not a controlled comparison - only a fixed archived trace is; driver and kernel changes may not be revertible independently of the engine.",
  "Rollback: if the previous build is clean on the fixed trace, roll back to it as the immediate mitigation while root-causing continues, provided the older build has no known correctness defect.",
  "Boundary condition: if both builds fail identically on the archived trace, the regression hypothesis is refuted and the cause is capacity or workload."),

 ("Graceful degradation: making the failure mode acceptable before making it rare",
  "Hypothesis H9 (falsifiable): the service can convert hard OOM (process crash, dropped in-flight requests) into a bounded, observable rejection with no crash. If true, after adding a memory-aware admission gate the failure manifests as HTTP 429/503 with zero engine restarts under the same trace.",
  ["Measure the current blast radius: how many in-flight requests are lost per OOM and whether the engine restarts; this determines urgency independently of root cause.",
   "Add a pre-admission check against free KV blocks with a configured safety margin, and emit a rejection metric; measure the rejection rate on the production trace before enforcing.",
   "Controlled experiment: identical trace with the gate in shadow mode and then in enforcing mode; compare restart count, dropped-request count and rejection rate."],
  "Confounders: a safety margin that is too large silently caps throughput and looks like a capacity shortage; retrying clients can amplify rejections into a load spiral.",
  "Rollback: if the rejection rate exceeds the agreed error budget or clients enter retry amplification, reduce the margin or disable enforcement within one deploy cycle.",
  "Boundary condition: degradation control does not reduce the underlying residency requirement; it only bounds harm, and must not be reported as a root-cause fix."),

 ("Decision record: which mitigation to keep and how to know it worked",
  "Hypothesis H10 (falsifiable): the chosen mitigation, whatever it is, produces a statistically distinguishable reduction in OOM rate on a reproducing baseline. If true, OOM count across three post-change replays is lower than the baseline's three-run min-max band with no overlap.",
  ["Require a reproduction gate first: at least 2 of 3 baseline replays must fail at a comparable position; without it no A/B is interpretable.",
   "Fix the metric set before the run: OOM count, restarts, p99 TTFT, p99 end-to-end latency, throughput, rejection rate. Post-hoc metric selection invalidates the conclusion.",
   "Controlled experiment: three replays per arm, one variable per arm, medians reported with min-max bands; overlapping bands mean 'not shown', not 'no effect'."],
  "Confounders: thermal state, background compaction, and page-cache warmth vary between early and late runs - randomise arm order rather than running all baselines first.",
  "Rollback: any mitigation whose bands overlap the baseline is reverted rather than kept 'just in case', since each retained knob raises the cost of the next investigation.",
  "Boundary condition: an improvement measured on a baseline that reproduces less than 2 of 3 times is not evidence at all and must be reported as inconclusive."),
]

RISKS = [
 ["Ranking mitigations on a single anecdotal failure leads to tuning the wrong subsystem.",
  "Coarse telemetry sampling aliases short spikes and makes unrelated failures look identical."],
 ["Token arithmetic that is never checked against measured occupancy becomes an unfalsifiable model.",
  "Prefix sharing and speculative decoding break the assumed token-to-bytes mapping."],
 ["Unbudgeted non-KV buffers make the KV pool fraction unsafe at startup.",
  "Disabling CUDA graphs relieves memory but degrades decode throughput."],
 ["Over-large max_tokens forces either pessimistic reservation or unsafe overcommit.",
  "Preemption hides OOM as latency, so OOM-count-only metrics can show false success."],
 ["Swap-based preemption can move the bottleneck to PCIe and regress throughput badly.",
  "Preemption thrashing produces recompute storms that look like a throughput bug."],
 ["Unverified device exclusivity invalidates all internal memory accounting.",
  "An idle co-tenant during reproduction makes the isolation arm falsely look like a fix."],
 ["Batch shape, not batch size, may drive peak transient memory; size-based limits then fail.",
  "Kernel path switches at length thresholds cause discontinuous workspace jumps."],
 ["Live A/B across time confounds code regressions with traffic drift.",
  "Rolling back an engine build may reintroduce a previously fixed defect."],
 ["A too-large admission safety margin silently caps throughput.",
  "Client retries can amplify rejections into a load spiral."],
 ["Post-hoc metric selection turns noise into a claimed improvement.",
  "Retaining ineffective knobs raises the cost of every future investigation."],
]

EVIDENCE = [
 ["Fixed per-failure telemetry record for at least five OOM events",
  "Cluster tightness/spread across those failures",
  "Instrumentation overhead measured as p99 latency delta"],
 ["per_token_KV_bytes derivation with all shape terms and units",
  "Summed in-flight token time series with its peak",
  "Predicted versus measured failure point on a concurrency ramp"],
 ["Post-warm-up memory attribution: weights, KV pool, captured graphs, residual",
  "Logits/sampling buffer size versus batch size and vocabulary",
  "Freed bytes, OOM count and decode latency with graph capture disabled"],
 ["Decode step index and remaining max_tokens for in-flight requests at failure",
  "Requested versus actually generated token length distributions",
  "Paired replays: pessimistic admission versus overcommit plus preemption"],
 ["Preemption / recompute / swap counters aligned to failure timestamps",
  "Re-prefilled tokens per preemption",
  "Throughput and p99 TTFT with preemption enabled and disabled"],
 ["Per-PID device memory time series at 1 s resolution",
  "MIG/MPS/co-location configuration verified explicitly",
  "Reproduction result on a verified-exclusive device"],
 ["Peak reserved bytes across at least four (num_sequences, length) points at fixed total tokens",
  "Attention kernel path selected at each length",
  "Throughput impact of any shape-aware admission rule"],
 ["Change inventory across the window when failures began",
  "Archived-trace replay results on previous and current builds",
  "Bisection results, only if the two-build comparison differs"],
 ["Dropped-request count and engine restart count per OOM",
  "Shadow-mode rejection rate on the production trace",
  "Restart, drop and rejection counts under enforcement"],
 ["Baseline reproduction rate (>= 2 of 3) with failure position",
  "Pre-registered metric set with medians and min-max bands per arm",
  "Randomised arm ordering record"],
]

CONF = [0.60, 0.63, 0.61, 0.60, 0.59, 0.58, 0.60, 0.62, 0.61, 0.63]
QD = [(2, 2, 2)] * 10


def build(k):
    title, hyp, steps, conf_txt, rb, bound = ANGLES[k]
    body = HDR
    body += "Primary lens for this case: %s.\n\n" % title
    body += "1) Falsifiable hypothesis\n   %s\n\n" % hyp
    body += "2) Prioritized diagnosis and measurements (cheapest and most discriminating first)\n"
    for i, s in enumerate(steps):
        body += "   %s. %s\n" % ("abc"[i], s)
    body += ("\n3) Controlled experiment design\n   One variable per arm, a byte-identical replayed request trace "
             "across arms, three repetitions per arm, medians reported with min-max bands. Any arm that moves two "
             "knobs at once is discarded rather than interpreted.\n\n")
    body += "4) Expected confounders\n   %s\n\n" % conf_txt
    body += "5) Boundary condition\n   %s\n\n" % bound
    body += ("6) Rollback criteria\n   %s Rollback executes within one deploy cycle and the restored configuration "
             "is re-verified by at least one replay.\n\n" % rb)
    body += ("7) What would refute this framing\n   If the allocator record at failure shows device free bytes near "
             "zero and a requested block larger than (reserved - allocated), the device is genuinely exhausted: only "
             "capacity, context limits, quantization or more GPUs can change the outcome, and every scheduling or "
             "allocator knob above is inert. Report that conclusion explicitly instead of continuing to tune.\n")
    return body


rows = []
with open(CORPUS) as f:
    for i, line in enumerate(f):
        if i < START:
            continue
        if i >= START + N:
            break
        d = json.loads(line)
        msgs = {m["role"]: m["content"] for m in d["messages"]}
        k = i - START
        tc, ic, os_ = QD[k]
        rows.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": msgs["user"],
            "source_assistant": msgs["assistant"],
            "corrected_answer": build(k),
            "quality_dimensions": {
                "technical_correctness": tc,
                "instruction_coverage": ic,
                "operational_safety": os_,
            },
            "risks": RISKS[k],
            "evidence_required": EVIDENCE[k],
            "confidence": CONF[k],
        })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), OUT)
