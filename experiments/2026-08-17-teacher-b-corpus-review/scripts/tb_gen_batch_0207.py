import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0207.jsonl"
START, N = 2060, 10

# Reuse the shared frame verbatim from an earlier batch generator.
src = open(f"{EXP}/scripts/tb_gen_batch_0205.py").read()
ns = {}
exec(src.split("STANCES = [")[0], ns)
COMMON = ns["COMMON"]

STANCES = [
 ("Stance 70 - Quantisation and weight dtype interact with the parallel axis, so the layout verdict is dtype-conditional.",
  "Sharding weights across TP ranks changes the size and shape of each rank's weight tiles, and quantised kernels have "
  "granularity constraints: group-wise scales, tile alignment and minimum in-features per rank. A TP degree that splits a "
  "quantised projection below the kernel's group granularity either falls back to a slower path or forces cross-rank scale "
  "handling, while the PP arm keeps whole projections intact on one device. A verdict measured in bf16 therefore does not "
  "carry to an int8 or fp8 deployment. Falsifiable hypothesis H70: for at least one TP degree that divides a projection's "
  "in-features below the quantisation group size, measured decode throughput in the quantised build is lower relative to "
  "the bf16 build than at a smaller TP degree, inverting the bf16 ranking (ESTIMATE; derivation: quantised kernel "
  "efficiency is a step function of per-rank in-features versus group size, so it does not scale with per-rank work). "
  "Controlled experiment: run the full layout sweep twice, once in bf16 and once in the deployment quantisation, holding "
  "traffic and engine version fixed, and log per-arm kernel selection and any dequantisation fallbacks. Evidence required: "
  "quantisation scheme and group size, per-rank in-features arithmetic for each TP degree, kernel and fallback logs, and "
  "paired bf16-versus-quantised results. Rollback gate: never ship a layout chosen under one weight dtype into a "
  "deployment using another without repeating the sweep in the shipping dtype.",
  ["A layout ranking measured in bf16 can invert under the quantisation actually deployed.",
   "TP degrees that break quantisation group granularity trigger silent slow-path fallbacks.",
   "Quantisation also perturbs output quality, so a layout change and a dtype change must not be evaluated together."],
  ["Quantisation scheme, group size and per-rank in-features arithmetic for every TP degree.",
   "Kernel dispatch and dequantisation-fallback logs per arm in the quantised build.",
   "Paired bf16 and quantised sweep results at identical traffic and engine version."]),

 ("Stance 71 - The scheduler's batching policy, not the parallel axis, is often the dominant latency term.",
  "Continuous batching, chunked prefill, preemption and the maximum-batched-tokens cap set how long a decode step waits "
  "behind other work. Those knobs typically move per-token latency by more than a one-step change in TP degree, and their "
  "defaults are not constant across layouts because memory headroom differs. Comparing layouts while the scheduler "
  "auto-tunes to each arm's free memory is comparing scheduler configurations. Falsifiable hypothesis H71: holding the "
  "layout fixed and sweeping only the maximum-batched-tokens and chunked-prefill settings produces a p95 TPOT spread at "
  "least as large as the spread across layouts at fixed scheduler settings (ESTIMATE; derivation: scheduler settings "
  "directly control queueing delay and prefill-decode interleaving, which are first-order additive terms in per-token "
  "latency, whereas the collective term changed by one TP step is a fraction of one step's compute). Controlled "
  "experiment: fix the layout, sweep scheduler settings alone and record the spread; then fix scheduler settings "
  "explicitly and sweep the layout, and compare the two spreads. Evidence required: the full effective scheduler "
  "configuration per arm as resolved at runtime rather than as written, queueing-delay attribution per request, and both "
  "sweep spreads. Rollback gate: reject any layout comparison in which the resolved scheduler configuration differed "
  "between arms.",
  ["Auto-tuned scheduler defaults differ per layout, silently confounding the comparison.",
   "The scheduler term can exceed the layout term, so the headline conclusion may be about the wrong variable.",
   "Written configuration and resolved runtime configuration diverge, so the intended pinning may not have taken effect."],
  ["The resolved runtime scheduler configuration per arm, dumped from the running engine.",
   "Per-request queueing-delay attribution separated from service time.",
   "Measured latency spread from a scheduler-only sweep at fixed layout, for comparison against the layout-only spread."]),

 ("Stance 72 - Warmup, memory fragmentation and capture state make the first minutes of every arm unusable as data.",
  "Engines allocate KV pools, autotune kernels, capture graphs and JIT-compile on first use. An arm measured immediately "
  "after start is measuring compilation and allocation, not steady state; and because capture and autotune cost differs by "
  "layout, the bias is not common-mode across arms. Fragmentation after long running additionally shifts achievable "
  "concurrency, so a late measurement and an early measurement of the same arm are different experiments. Falsifiable "
  "hypothesis H72: discarding a pre-declared warmup window changes the inter-layout gap by more than the run-to-run "
  "variation measured within a single arm, which would show that warmup contamination was material (ESTIMATE; derivation: "
  "compilation, graph capture and allocator growth are one-time costs concentrated at start, so their inclusion biases "
  "early windows by an arm-dependent amount that is not present in steady state). Controlled experiment: declare a warmup "
  "window and a steady-state window before running, report both, and additionally repeat each arm from a cold process at "
  "least three times to bound run-to-run variance. Evidence required: timestamped throughput and latency traces covering "
  "the whole run including warmup, allocator and fragmentation statistics at each window, graph-capture and compilation "
  "logs, and at least three cold repeats per arm. Rollback gate: discard any comparison whose inter-arm gap is smaller "
  "than the measured within-arm cold-restart variance.",
  ["Early-window measurements capture compilation and allocation rather than steady-state serving.",
   "Warmup cost differs by layout, so the contamination is not common-mode and does not cancel.",
   "A single run per arm cannot distinguish a real gap from cold-restart variance."],
  ["Full timestamped traces including the warmup window, with the window boundary pre-declared.",
   "Allocator and fragmentation statistics sampled at each declared window.",
   "At least three cold-process repeats per arm with reported within-arm variance."]),

 ("Stance 73 - Failure-domain sizing, not steady-state latency, should set the upper bound on parallel degree.",
  "Every device added to a tightly coupled group enlarges the blast radius: with TP, one failed rank kills the whole "
  "replica, and with cross-node PP a single node failure stalls the pipeline. Availability therefore degrades roughly "
  "with the number of components whose failure takes down one serving unit, while latency improves at best sublinearly "
  "with parallel degree. The correct framing sets a maximum acceptable failure domain from the availability budget first, "
  "then optimises latency only within layouts that satisfy it. Falsifiable hypothesis H73: measured replica-level "
  "unavailability increases monotonically with the number of devices per replica, and at the largest degree under test it "
  "exceeds the error budget even though that degree wins on steady-state latency (ESTIMATE; derivation: for independent "
  "component failures the probability that a serving unit contains at least one failed component rises with the component "
  "count, while parallel speedup is bounded by communication overhead). Controlled experiment: inject single-rank and "
  "single-node faults at each parallel degree, measure detection time, replica recovery time and requests lost, and "
  "convert to an availability figure comparable against the error budget. Evidence required: per-device and per-node "
  "failure rates from fleet history, fault-injection recovery timings per degree, the stated availability budget, and "
  "requests-lost accounting. Rollback gate: exclude any layout whose modelled replica availability breaches the error "
  "budget, regardless of its latency result.",
  ["Larger parallel groups enlarge the failure domain faster than they improve latency.",
   "Steady-state benchmarks contain no availability information at all, so a latency-only choice is underspecified.",
   "Recovery time, not just failure probability, determines whether the error budget is breached."],
  ["Fleet-derived per-device and per-node failure rates rather than vendor nominal figures.",
   "Fault-injection detection and recovery timings measured at each parallel degree.",
   "The service availability budget and requests-lost accounting per injected fault."]),

 ("Stance 74 - Cost per delivered token at the target latency, not latency alone, is the decision metric.",
  "A layout that lowers per-token latency while lowering achievable concurrency can raise cost per token, and a service "
  "budget is denominated in cost at an SLO, not in latency at any occupancy. The comparison that answers the actual "
  "business question fixes the latency SLO, measures the maximum sustained throughput each layout achieves while holding "
  "that SLO, and divides device-hours by delivered tokens. Layouts frequently reverse rank between the latency view and "
  "the cost-at-SLO view. Falsifiable hypothesis H74: the layout with the best unloaded per-token latency is not the "
  "layout with the lowest cost per delivered token at the SLO, because it saturates at lower concurrency (ESTIMATE; "
  "derivation: collective overhead per step is paid at every concurrency level while KV-capacity limits set the "
  "concurrency ceiling, so the layout optimising the former can lose on the latter). Controlled experiment: for each "
  "layout, sweep offered load to find the maximum rate sustaining the SLO, then compute device-hours per million "
  "delivered tokens at that point. Evidence required: the written latency SLO and its percentile, load-sweep curves to "
  "saturation per layout, device-hour accounting including idle and redundant capacity, and delivered-token counts. "
  "Rollback gate: do not adopt a layout on an unloaded latency result; require the cost-at-SLO computation before any "
  "capacity commitment.",
  ["Unloaded latency comparisons omit the concurrency ceiling and therefore omit cost.",
   "Layout rankings can reverse between the latency view and the cost-at-SLO view.",
   "Device-hour accounting that ignores redundancy and idle capacity understates true cost asymmetrically."],
  ["The written latency SLO including the percentile it is defined at.",
   "Offered-load sweeps to saturation per layout with the SLO-sustaining rate identified.",
   "Device-hour accounting including redundancy and idle capacity, paired with delivered-token counts."]),

 ("Stance 75 - Statistical discipline: declare the estimator and the required sample size before collecting tail latency.",
  "Tail percentiles are noisy estimators with heavy-tailed error, and p99 from a short run has a confidence interval wide "
  "enough to contain most plausible layout gaps. Reporting a point p99 per arm and picking the smaller is not a "
  "comparison, it is a coin flip dressed as measurement. The disciplined form pre-declares the percentile, the aggregation "
  "window, the number of independent repeats, and the interval method, then checks whether the intervals separate. "
  "Falsifiable hypothesis H75: bootstrap confidence intervals on p99 from the planned run length overlap between the two "
  "leading layouts, meaning the headline ranking is not statistically supported at the planned sample size (ESTIMATE; "
  "derivation: the variance of an extreme quantile estimator falls only with the number of samples in the tail region, so "
  "short runs yield intervals whose width is comparable to typical inter-layout differences). Controlled experiment: "
  "pre-register the percentile, window, repeat count and interval method; collect the planned samples; report intervals "
  "rather than points and state explicitly when they overlap. Evidence required: raw per-request latency samples rather "
  "than pre-aggregated summaries, the pre-registration of estimator and sample size, the repeat structure, and computed "
  "intervals. Rollback gate: report 'no distinguishable difference' whenever the intervals overlap, and never break the "
  "tie with a secondary metric chosen after seeing the data.",
  ["Point-estimate tail percentiles from short runs cannot support a ranking.",
   "Choosing the deciding metric after seeing the data manufactures significance.",
   "Pre-aggregated summaries destroy the raw samples needed to compute honest intervals."],
  ["Raw per-request latency samples retained for every arm and repeat.",
   "A pre-registration fixing percentile, aggregation window, repeat count and interval method.",
   "Computed confidence intervals with an explicit statement of whether they separate."]),

 ("Stance 76 - Multi-tenant noisy neighbours and placement make single-tenant results non-transferable.",
  "A benchmark on an idle cluster measures a machine that will never exist in production. Co-tenants contend for host "
  "memory bandwidth, PCIe, NIC queues and network fabric, and a cross-node layout is exposed to fabric contention that an "
  "intra-node layout is not. Scheduler placement also varies between deployments, so the same layout can land on "
  "different topologies on different days. Falsifiable hypothesis H76: introducing a controlled background load on the "
  "shared fabric degrades the cross-node arm's tail latency substantially more than the intra-node arm's, narrowing or "
  "reversing the idle-cluster ranking (ESTIMATE; derivation: the cross-node arm's per-step critical path includes shared "
  "fabric transit whose queueing delay grows with aggregate offered load, a term the intra-node arm does not have). "
  "Controlled experiment: repeat the headline comparison with a calibrated background load on the fabric and on host "
  "resources, and additionally pin placement so topology is identical across repeats. Evidence required: the placement "
  "and topology actually realised for each arm, background-load calibration, fabric queue and congestion counters during "
  "the run, and paired idle-versus-loaded results. Rollback gate: do not extrapolate an idle-cluster ranking to a shared "
  "cluster; re-measure under representative contention before committing capacity.",
  ["Idle-cluster benchmarks systematically favour layouts that are sensitive to shared-fabric contention.",
   "Placement varies between runs, so topology is an uncontrolled variable unless explicitly pinned.",
   "Host-level contention on memory bandwidth and PCIe affects arms unequally and is rarely instrumented."],
  ["The realised placement and topology for every arm, recorded per run.",
   "Calibration of the injected background load and fabric congestion counters during the run.",
   "Paired idle-cluster and loaded-cluster results for the same comparison."]),

 ("Stance 77 - Engine and library versions must be pinned and recorded, because the answer expires.",
  "Layout performance is a property of a specific engine build, communication library version, driver and kernel stack. "
  "Collective algorithm selection, attention kernels and scheduler heuristics change between releases, and a conclusion "
  "recorded without its version context becomes folklore that outlives its validity and is re-cited years later. Two "
  "arms accidentally built on different versions produce a version comparison labelled as a layout comparison. "
  "Falsifiable hypothesis H77: re-running the identical comparison on a later engine and communication-library release "
  "changes the inter-layout gap by more than the within-arm run-to-run variance, which would establish that the "
  "conclusion is version-scoped rather than structural (ESTIMATE; derivation: releases change collective algorithm "
  "selection and kernel dispatch, both of which sit on the measured critical path, so their effect is not bounded by "
  "measurement noise). Controlled experiment: record the full software bill of materials per arm, verify the arms are "
  "identical except for layout, and schedule a repeat on the next release to test durability. Evidence required: engine, "
  "communication library, driver and kernel versions per arm; a diff confirming the arms match except for layout; and "
  "results from at least two software baselines. Rollback gate: mark any layout conclusion as expired when the engine or "
  "communication library major version changes, and re-run before citing it again.",
  ["Layout conclusions are version-scoped and silently expire after upgrades.",
   "Accidentally mismatched arms turn a version difference into a fake layout finding.",
   "Undated conclusions get re-cited long after the stack they were measured on is gone."],
  ["A full software bill of materials per arm covering engine, communication library, driver and kernel.",
   "A verified diff showing the arms are identical except for the parallel layout.",
   "Repeat results on at least two software baselines with dates attached."]),

 ("Stance 78 - Instrument the collective layer directly rather than inferring communication cost from end-to-end latency.",
  "Attributing a latency gap to communication without measuring communication is an inference, not a finding. The "
  "collective layer exposes which algorithm and protocol were selected, which transport was used, and how much time each "
  "rank spent waiting; that wait time also contains load imbalance and stragglers, which are not communication cost even "
  "though they appear at the collective. Without per-rank timing, a slow rank and a slow network are indistinguishable. "
  "Falsifiable hypothesis H78: a substantial share of the measured collective wait time is straggler-induced imbalance "
  "rather than transfer time, shown by per-rank arrival timestamps at the collective being dispersed rather than "
  "simultaneous (ESTIMATE; derivation: a collective completes no earlier than its last participant arrives, so arrival "
  "dispersion converts imbalance into apparent communication time). Controlled experiment: capture per-rank arrival and "
  "completion timestamps for the decode-path collectives, decompose wait time into dispersion and transfer, and "
  "separately run a standalone collective bandwidth test on the same ranks for a transfer-time reference. Evidence "
  "required: per-rank collective arrival and completion timestamps, communication-library algorithm and transport "
  "selection logs, standalone collective benchmark results on the same topology, and per-rank device clock and power "
  "state. Rollback gate: do not attribute a latency gap to interconnect bandwidth until arrival dispersion has been "
  "measured and excluded.",
  ["Collective wait time conflates transfer cost with straggler-induced load imbalance.",
   "A single slow or throttled rank presents as a network problem and misdirects the whole investigation.",
   "Algorithm and transport selection can differ between arms without any configuration change."],
  ["Per-rank arrival and completion timestamps for decode-path collectives.",
   "Communication-library algorithm, protocol and transport selection logs per arm.",
   "Standalone collective bandwidth results on the same ranks and topology, plus per-rank clock and power state."]),

 ("Stance 79 - Close by naming what this record is and the authority it does not carry.",
  "This is a provisional single-lane rewrite written without executing any benchmark on the asker's hardware. Everything "
  "above is mechanism, boundary conditions and experimental design; no number here is MEASURED, and every quantitative "
  "claim is tagged ESTIMATE with its derivation attached so a reader checks the reasoning instead of trusting the figure. "
  "The source pair is degenerate - the assistant turn is a grading rubric rather than an answer - so there is no prior "
  "answer to agree or disagree with, and no second party has corroborated this one. Falsifiable hypothesis H79: records "
  "carrying an explicit authority bound are cited as ground truth by downstream training or evaluation steps at a lower "
  "rate than otherwise identical records without one, which would show the label functions rather than decorates "
  "(ESTIMATE; derivation: absent an explicit bound, structured confident prose is read as authoritative regardless of "
  "provenance). Controlled experiment: attach this provenance block to every record, then audit downstream usage for any "
  "step that treated these as gold. Evidence required: the lane-isolation audit trail, a pre-declared inter-lane "
  "agreement metric computed only after both lanes are frozen, and a register of downstream consumers. Rollback gate: no "
  "training, evaluation or capability claim may cite these records as ground truth before an inter-lane agreement result "
  "exists and is published with them.",
  ["Provisional single-lane review is routinely over-read as adjudicated ground truth downstream.",
   "Confident structured prose signals an authority the provenance does not support.",
   "Without a lane-isolation audit, apparent agreement may reflect contamination rather than convergence."],
  ["An audit trail showing teacher lanes were isolated during generation.",
   "A pre-declared inter-lane agreement metric computed only after both lanes are frozen.",
   "A register of every downstream consumer of these records and how they were used."]),
]

CRITIQUE = (
"Critique of the source item: the prompt is a legitimate infrastructure question and does ask for assumptions, a "
"falsifiable hypothesis, measurements, confounders and rollback criteria, but the corpus pair is degenerate - the "
"assistant turn contains only a rubric describing what an answer should contain, not an answer. There is therefore no "
"substantive content to keep, and the item is rewritten into a complete response that supplies the mechanism, the "
"boundary conditions that flip the recommendation, an explicit falsifiable hypothesis, a single-variable controlled "
"experiment, the evidence artifacts required to adjudicate it, and a rollback gate. Every quantitative claim is labelled "
"ESTIMATE and carries its derivation; no value here is MEASURED, because no benchmark run was performed for this review. "
"This output is provisional teacher-B review material, not expert gold, and it is not evidence about any model's domain "
"capability."
)


def main():
    with open(CORPUS) as f:
        lines = f.readlines()[START:START + N]
    assert len(lines) == N, len(lines)
    assert len(STANCES) == N
    out = []
    for i, line in enumerate(lines):
        d = json.loads(line)
        m = {x["role"]: x["content"] for x in d["messages"]}
        su, sa = m["user"], m["assistant"]
        title, body, risks, ev = STANCES[i]
        ca = f"Analytical stance under test: {title}\n\n{COMMON}\n{body}\n\n{CRITIQUE}"
        out.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": ca,
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 2,
            },
            "risks": [
                "Source pair is degenerate: the assistant turn is a grading rubric rather than an answer.",
                "A bare TP-versus-PP verdict without interconnect, context length and concurrency context is not decidable.",
            ] + risks,
            "evidence_required": [
                "Interconnect topology dump and NCCL transport selection log for every arm of the comparison.",
                "Concurrency-resolved p50/p95/p99 TTFT and TPOT curves rather than mean end-to-end latency.",
            ] + ev,
            "confidence": 0.62,
        })
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out))


main()
