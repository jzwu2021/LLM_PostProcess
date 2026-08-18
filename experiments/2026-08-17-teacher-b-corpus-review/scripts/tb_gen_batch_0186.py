import ast, json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT = f"{EXP}/results/train-batch-0186.jsonl"
START = 1850
N = 10

src = open(f"{EXP}/scripts/tb_gen_batch_0182.py").read()
tree = ast.parse(src)
COMMON_TAIL = None
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "COMMON_TAIL":
        COMMON_TAIL = ast.literal_eval(node.value)
assert COMMON_TAIL and len(COMMON_TAIL) > 2000

rows = [json.loads(l) for l in open(CORPUS) if l.strip()]
sel = rows[START:START + N]
assert len(sel) == N

STANCES = [
    ("Analytical stance under test: calibration-set-first - the quantized model is a function of the data you calibrated on.",
     "Falsifiable hypothesis H31: swapping the calibration set for a same-size sample drawn from a different domain moves at least one per-slice quality metric by more than the pre-registered indifference threshold. If quality is invariant to calibration provenance, calibration is not the sensitive knob and effort should move elsewhere; if it is sensitive, no result gathered on an undocumented calibration set is transferable.",
     "Treat the calibration corpus as a first-class experimental input with its own hash, size, sequence-length distribution and domain mixture, not as a script default. Weight-only schemes fit per-group scales to activation statistics observed during calibration, so a calibration mixture that omits long-context, structured-output or non-English traffic will produce clipping behaviour that only fails on exactly those slices in production. Run the calibration-provenance ablation explicitly with at least two disjoint sets and report per-slice deltas, because a single averaged score hides a domain-localised regression. ESTIMATE: sensitivity typically grows as bit width falls and as group size grows, since fewer scale parameters must cover a wider dynamic range; derivation is the parameter-count-versus-dynamic-range argument only, and it is an ESTIMATE, not MEASURED. Record calibration sequence length, because activation outlier magnitude in attention projections varies with context length and a short-sequence calibration can silently misfit long-context serving. Rollback gate: if the two calibration arms disagree beyond threshold on any shipped slice, the recipe is not qualified regardless of throughput."),

    ("Analytical stance under test: outlier-and-numerics-first - the failure mode is a few channels, not the average error.",
     "Falsifiable hypothesis H32: per-slice quality damage correlates with per-layer maximum activation-channel magnitude rather than with mean weight quantization error, verified by ranking layers on both statistics and comparing against a layer-wise sensitivity sweep. If mean error predicts damage as well as outlier magnitude does, the outlier-handling machinery is unnecessary complexity and should be removed.",
     "Quantization damage is concentrated, not diffuse: a small number of activation channels with large magnitude dominate the error a group-wise scale must absorb, which is why per-channel or per-group schemes, symmetric versus asymmetric choices, and outlier-preserving mixed-precision paths change results far more than the nominal bit width suggests. Measure rather than assume: run a layer-wise sensitivity sweep that keeps one layer in BF16 at a time and record the per-slice quality recovery, then compare that ranking against both mean weight error and activation-outlier magnitude. Report which layers you excluded from quantization and why, since partial coverage changes the byte ratio that bounds any speedup claim and is the most common reason a published number fails to reproduce. Also record whether accumulation is in FP32 and whether the scale dtype differs between arms; an accumulator or scale-dtype difference is a confound that masquerades as a quantization quality result. Any numeric claim here must carry an ESTIMATE or MEASURED label with its derivation."),

    ("Analytical stance under test: null-result-discipline-first - design the experiment so that no benefit is a publishable outcome.",
     "Falsifiable hypothesis H33: the pre-registered analysis, applied mechanically, returns no significant cost-per-token improvement at the SLO. Pre-committing to publish and act on that outcome is what makes the positive outcome credible; if the team cannot state in advance what a null looks like, the experiment is not controlled.",
     "State before data collection what evidence would cause you to abandon weight-only quantization for this checkpoint entirely, not merely what would cause adoption. Define the null concretely: cost-per-token improvement inside the A/A noise band, or improvement that does not remove an integer number of replicas at the unchanged headroom policy, or improvement that is real but paid for by a per-slice quality delta exceeding the indifference threshold. Pre-register slices, stopping rule and sample size so post-hoc slice selection cannot manufacture a win; the most common failure in serving-cost experiments is not fabrication but drift - the team keeps looking until a favourable cut appears. Require the write-up to separate MEASURED with intervals and artifact references, ESTIMATE with inline derivation, and ASSUMED with source, and ban unlabelled numbers. Record the null in the same artifact store as a positive result so the next team does not silently repeat the work, and attach an expiry tied to the next engine or checkpoint bump, since a null under one kernel generation can invert under the next."),

    ("Analytical stance under test: engine-scheduler-interaction-first - quantization changes batch composition, and batch composition sets throughput.",
     "Falsifiable hypothesis H34: with the scheduler's max-batched-tokens, max-num-seqs, chunked-prefill and preemption policy pinned identically across arms, the INT4 throughput gain differs materially from the gain measured under each arm's auto-tuned scheduler settings. If the two agree, scheduler interaction is not a confound here; if they diverge, no single-number speedup is meaningful without naming the scheduler regime.",
     "A serving engine's scheduler is part of the system under test. Smaller weights change how many sequences fit, which changes batch composition, which changes the prefill-decode interleave, which changes both TTFT and TPOT in ways that do not follow from the weight-byte ratio at all. Run the comparison twice - once with every scheduler knob pinned identically so the numeric path is the only difference, and once with each arm auto-tuned so you measure the deployable configuration - and report both with the mechanism named. Record per-step batch-size and token-count histograms for both arms; if the histograms differ, throughput differences are partly a scheduling result and must be attributed as such. Watch preemption and recompute events specifically, since an arm with more KV headroom preempts less and that alone can dominate the measured gain. Also confirm chunked-prefill settings match, because prefill chunking changes the TTFT-TPOT tradeoff independently of quantization and is frequently left at differing defaults between arms."),

    ("Analytical stance under test: tail-latency-first - the mean is a marketing number, the tail is the SLO.",
     "Falsifiable hypothesis H35: the p99 TTFT and p99 TPOT of the INT4 arm are no worse than the BF16 arm at matched offered load, measured client-side including queueing. If p50 improves while p99 degrades, the change is a regression in SLO terms even though the headline throughput number improved.",
     "Read the distribution, not the average. Quantization can improve mean decode throughput while worsening the tail through kernel-selection variance, occasional reference-path fallback on unusual shapes, or increased queueing caused by different admission behaviour at higher concurrency. Measure at matched offered load rather than matched concurrency, report p50, p95 and p99 for TTFT and TPOT separately, and take timestamps at the client so queueing is included; server-side-only timing systematically hides the exact regression that pages an on-call engineer. Exclude warmup, compilation and cache-fill windows with a stated rule, and record cancelled and truncated requests separately, since a system that sheds load can appear faster in surviving-request statistics. Plot the full latency-throughput frontier for both arms because arms commonly cross, and read the comparison at the SLO percentile rather than at saturation. Any capacity conclusion must name the percentile, the window and the measurement point, or it is not a capacity conclusion."),

    ("Analytical stance under test: multi-GPU-topology-first - the parallel layout can erase or invent the benefit.",
     "Falsifiable hypothesis H36: the quantization benefit measured at the deployed tensor-parallel degree differs materially from the benefit measured at TP=1 on the same hardware. If it does, per-GPU weight-byte reduction is interacting with collective cost and the single-GPU result must not be used for fleet sizing.",
     "Under tensor parallelism the per-GPU weight footprint shrinks with TP degree, so the fraction of decode time spent reading weights falls while all-reduce cost per token stays roughly fixed, which mechanically compresses the headroom quantization can recover. Measure at the TP degree you will actually deploy, and at TP=1 as a reference, and state both. Record NCCL algorithm and protocol selection, the interconnect in use, and whether the all-reduce is overlapped with compute, since an arm that happens to pick a different collective algorithm is not a clean comparison. Confirm both arms use identical TP and pipeline layout and identical placement on the physical topology, because crossing a slower link in one arm is an invisible confound that looks like a quantization effect. ESTIMATE: as TP degree rises the achievable decode speedup from weight-only quantization declines toward the collective-bound floor; derivation is the shrinking weight-byte share of per-token time under fixed collective cost, and it is an ESTIMATE, not MEASURED. Verify by measuring at both endpoints rather than extrapolating."),

    ("Analytical stance under test: structured-output-and-parser-contract-first - the downstream consumer is part of the SLO.",
     "Falsifiable hypothesis H37: structured-output compliance rate - valid JSON, schema conformance, tool-call arity and argument types - is unchanged between arms within a pre-registered threshold on a fixed panel. If compliance drops while aggregate quality scores hold, the aggregate metric is not measuring what production depends on and must be replaced.",
     "Downstream systems parse the output, and a format regression breaks them even when semantic quality is unchanged. Quantization noise perturbs low-margin token decisions, which is exactly where delimiter, brace and schema-key tokens live, so constrained decoding and tool-call formatting are disproportionately exposed. Build a fixed compliance panel with machine-checkable pass conditions and run it as a standing canary during rollout, not only once before it. Hold sampling parameters, seeds, stop conditions and any grammar or constrained-decoding backend identical across arms and record them, because a grammar-backend difference will dominate any quantization effect. Measure output-length distribution as well, since truncation and runaway generation both show up there before they show up in a quality score. Retain raw generations for both arms so a disputed compliance regression can be adjudicated from evidence. Abort threshold on compliance rate must be written into the change record before the ramp begins, and breaching it triggers rollback regardless of throughput results."),

    ("Analytical stance under test: capacity-realisation-first - a throughput gain that never removes a replica saved nothing.",
     "Falsifiable hypothesis H38: after rollout, matched-window fleet accounting shows GPU-hours per million served tokens fell by at least the predicted amount. If serving-hours are flat while benchmark throughput rose, the gain was absorbed by headroom policy, autoscaler behaviour or fragmentation, and the predicted saving was not realised.",
     "Close the loop between benchmark and fleet. Convert the throughput result into a replica-count prediction under the unchanged headroom and autoscaling policy, state that prediction numerically before rollout, then reconcile against measured fleet GPU-hours, served tokens and replica counts over matched windows afterwards. Name the mechanisms that commonly absorb gains: minimum-replica floors, per-zone redundancy, scale-down cooldowns, memory fragmentation preventing the extra KV blocks from being usable, and traffic growth coinciding with the ramp. Require the reconciliation to be a signed obligation with an owner and a date, because unreconciled savings claims accumulate into planning error that is discovered only at budget time. If the prediction depended on freed HBM raising concurrency, verify the concurrency actually rose in production rather than assuming the benchmark condition held. Publish the reconciliation whether or not it confirms the prediction, and label every figure MEASURED with its artifact reference or ESTIMATE with its derivation."),

    ("Analytical stance under test: dual-path-operational-burden-first - the second numeric path is a permanent liability, not a one-off cost.",
     "Falsifiable hypothesis H39: the team can demonstrate, by rehearsal, that a fresh on-call engineer can determine which numeric path served a given request and roll it back within the stated time-to-safe, using only production tooling. If that rehearsal fails, the dual-path configuration is not operable and should not be shipped.",
     "Adopting quantization means running two numeric paths across every future checkpoint, engine bump and driver upgrade, and that recurring burden is usually omitted from the cost model that justified the change. Make numeric path an explicit slicing dimension on every dashboard, alert, log line and trace attribute so an incident can implicate or exonerate it in minutes rather than hours. Require per-checkpoint re-qualification as a gate, not a best-effort follow-up, and state who owns it and what happens when it is skipped. Keep the BF16 path warm and routable throughout, and rehearse the flip with a stopwatch so time-to-safe is MEASURED rather than asserted. Include in the cost model the calibration and quantization runs, their failure rate, the extra eval compute, the engineering hours, and the incident risk premium, then amortise over the small number of checkpoints the recipe will actually serve - that amortisation term is the one most often left out and is frequently large enough to flip the decision. If the burden cannot be staffed, the correct outcome is to decline the change even with a genuine throughput win."),

    ("Analytical stance under test: alternative-hypothesis-first - compare against the cheaper interventions before adopting a numeric change.",
     "Falsifiable hypothesis H40: weight-only quantization delivers a larger cost-per-token improvement at the SLO than the best non-numeric intervention - scheduler tuning, chunked-prefill configuration, KV-cache policy, prefix caching, or right-sizing the parallel layout - measured on the same trace and harness. If a cheaper intervention matches or beats it, quantization is not the correct next step regardless of its absolute gain.",
     "The relevant comparison is not INT4 versus BF16 in isolation but INT4 versus the alternative uses of the same engineering effort, each evaluated on the identical trace, SLO and harness. Enumerate the candidates explicitly and measure at least the strongest one or two, because scheduler and cache-policy wins are often comparable in magnitude, carry no quality risk, and impose no dual-path burden. Report all arms on one latency-throughput frontier so the comparison is visible rather than argued. Where interventions compose, state whether you measured them independently or stacked, since stacked gains are routinely sub-additive when they contend for the same bottleneck. Pre-register the decision rule so the ranking, not the narrative, selects the winner. State the generalisation boundary of the conclusion: model family, bit width, engine version, accelerator generation, context-length range and batch-size range, and require re-qualification outside it. Every quantitative claim carries an ESTIMATE or MEASURED label with derivation or artifact reference."),
]

out = []
for row, (stance, hyp, body) in zip(sel, STANCES):
    m = {x["role"]: x["content"] for x in row["messages"]}
    ans = f"{stance}\n\n{hyp}\n\n{body}{COMMON_TAIL}"
    out.append({
        "source_id": row["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": ans,
        "quality_dimensions": {
            "technical_correctness": 3,
            "instruction_coverage": 2,
            "operational_safety": 3,
        },
        "risks": [
            "source_assistant is a grading rubric, not an answer; training on it teaches the model to restate evaluation criteria instead of performing the analysis",
            "rubric never names calibration-set provenance or activation-outlier concentration, so a model can claim a quality result that is an artifact of an undocumented calibration mixture",
            "rubric does not require pinning scheduler and parallel-layout settings across arms, permitting a batching or collective-topology effect to be reported as a quantization effect",
            "no requirement to measure structured-output and tool-call compliance, so a format regression that breaks downstream parsers passes an aggregate quality gate",
            "no ESTIMATE-versus-MEASURED labelling and no null-result pre-registration, so post-hoc slice selection can manufacture a favourable outcome",
            "no fleet reconciliation obligation, so a benchmark gain absorbed by headroom policy is recorded as a realised saving",
            "no canary, abort threshold or rehearsed time-to-safe, so an unsafe fleet-wide numeric-path change is not excluded",
        ],
        "evidence_required": [
            "per-request raw records: arrival, admission, first-token-to-client, per-token timestamps, output length, terminal status, taken client-side",
            "arm configuration manifests with checkpoint, calibration-set, engine, library and driver hashes showing a bit-width-only symmetric difference",
            "calibration-provenance ablation across two disjoint calibration sets with per-slice deltas",
            "layer-wise sensitivity sweep plus per-layer kernel name and achieved-bandwidth traces, excluding silent dequant-GEMM fallback and documenting partial quantization coverage",
            "per-step batch-size, token-count, preemption and recompute histograms for both arms under pinned and auto-tuned scheduler settings",
            "TP=1 and deployed-TP measurements with NCCL algorithm, protocol, interconnect and placement recorded",
            "per-slice quality scores with bootstrap 95% CIs on absolute deltas, structured-output and tool-call compliance panel results, retained raw generations, and a BF16-vs-BF16 A/A noise floor",
            "replayed production arrival trace with the full latency-throughput frontier at the stated SLO percentile, p50/p95/p99 TTFT and TPOT",
            "rehearsed rollback record with measured time-to-safe, and matched-window fleet GPU-hour, token and replica-count reconciliation confirming or refuting realisation",
        ],
        "confidence": 0.62,
    })

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("WROTE", OUT, len(out))
print("IDS", ",".join(r["source_id"] for r in out))
print("OPENINGS_DISTINCT", len({r["corrected_answer"][:200] for r in out}))
