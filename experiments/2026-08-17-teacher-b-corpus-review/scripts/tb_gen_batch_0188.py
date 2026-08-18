import ast, json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT = f"{EXP}/results/train-batch-0188.jsonl"
START = 1870
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
    ("Analytical stance under test: memory-bandwidth-roofline-first - predict the ceiling before measuring, then explain the gap.",
     "Falsifiable hypothesis H41: the measured decode speedup does not exceed the weight-byte ratio between arms at matched batch size, and where it falls short the deficit is attributable to a named non-weight term - KV-cache traffic, dequantization overhead, collective time or kernel launch gaps - identified in a profile. If the measured speedup exceeds the byte ratio, some other difference between arms is uncontrolled and the comparison is invalid.",
     "Write the roofline prediction down before running anything. Decode at small batch is weight-bandwidth-bound, so the first-order ceiling on speedup is the ratio of bytes read per token between arms, adjusted for the fact that KV-cache reads, activations and any dequantization traffic are not reduced by weight-only quantization. ESTIMATE: at batch sizes where KV traffic is a large share of bytes moved, realised speedup falls well below the nominal weight-byte ratio; derivation is Amdahl applied to the unquantized byte terms, and it is an ESTIMATE, not MEASURED. Then measure and reconcile: a result far below prediction usually means dequant is not fused into the GEMM or the kernel fell back to a reference path, and a result above prediction means an uncontrolled difference such as a different batch size, sequence length, engine version or cache-hit rate. Require the profile that names the executed kernels and their achieved bandwidth, because a speedup claim without a kernel-level attribution cannot be defended when it fails to reproduce on the next engine bump. Rollback gate: no adoption while the measured-versus-predicted gap is unexplained."),

    ("Analytical stance under test: KV-cache-accounting-first - weight-only quantization does not touch the term that usually binds at scale.",
     "Falsifiable hypothesis H42: freed HBM from smaller weights converts into an increase in usable KV blocks and sustained concurrency of at least the predicted amount at the same SLO. If concurrency does not rise, the memory saving did not become capacity and no cost-per-token improvement should be claimed from it.",
     "Separate the two mechanisms quantization is credited with: fewer weight bytes read per decode step, and more HBM left over for KV cache. They have different failure modes and must be measured separately. Compute the predicted KV block count from the freed bytes, the block size and the reserved fraction, state it numerically as an ESTIMATE with its derivation, then measure actual sustained concurrency at the SLO. Common reasons the conversion fails: the memory-utilisation fraction was left at a default that reclaims the savings as slack, fragmentation prevents allocating whole blocks, the workload is prefill-heavy so decode concurrency was never the binding constraint, or an admission-control limit caps concurrency below what memory now permits. Record max-model-len, block size, preemption and swap counts for both arms, since an arm that stops preempting gains throughput for a reason that has nothing to do with numerics. If the deployment is long-context, state explicitly that KV bytes dominate weight bytes and that the expected benefit is therefore small, rather than quoting a short-context benchmark."),

    ("Analytical stance under test: reproducibility-and-provenance-first - a result that cannot be re-run is not a result.",
     "Falsifiable hypothesis H43: an independent engineer, given only the recorded artifacts, reproduces both arms' headline numbers within the pre-registered noise band without consulting the original author. If reproduction fails, the difference between the two runs is an uncontrolled variable that was also present in the original comparison.",
     "Treat reproduction as a gate, not a courtesy. Record for each arm: checkpoint hash, quantization recipe and its exact parameters, calibration-set hash, engine and kernel-library versions, driver and firmware versions, container image digest, hardware SKU and placement, environment variables that affect kernel or collective selection, sampling parameters and seeds, and the harness commit. Pin the trace file and the eval panel by hash so the input is not silently regenerated. Run the reproduction on different physical hosts of the same SKU to expose host-level confounds such as thermal state, clock policy, NUMA placement or a mismatched driver, which are among the most common causes of an unreproducible serving number. Include a BF16-versus-BF16 A/A run in the reproduction package so the reader can see the noise floor the claimed delta must clear. If any element cannot be pinned, name it explicitly as an ASSUMED variable with its suspected influence rather than omitting it. Rollback gate: a headline number that fails independent reproduction is withdrawn, not footnoted."),

    ("Analytical stance under test: workload-representativeness-first - the benchmark trace decides the answer.",
     "Falsifiable hypothesis H44: the cost-per-token improvement measured on a replayed production arrival trace matches, within the pre-registered threshold, the improvement measured on the synthetic fixed-length benchmark. If they diverge, the synthetic number is not a deployment prediction and must not be used for sizing.",
     "Fixed input and output lengths at fixed concurrency are a kernel microbenchmark wearing serving clothes. Real traffic has a heavy-tailed input-length distribution, a different output-length distribution, bursty arrivals, prefix-cache hits and cancellations, and each of those changes the prefill-decode mix that determines how much weight-bandwidth relief is worth. Replay a real arrival trace with real length distributions, and report the input and output length percentiles alongside the result so a reader can judge transfer. State the prefix-cache hit rate for both arms and hold it constant, since a warmer cache in one arm shifts work out of prefill and manufactures an apparent decode win. If the deployment serves multiple traffic classes, report per-class results, because an aggregate that mixes short interactive requests with long batch requests can show a net gain while the interactive class regresses. ESTIMATE: benefit from weight-only quantization is largest for decode-heavy, short-prompt, high-concurrency traffic and smallest for prefill-heavy long-prompt traffic; derivation is the relative share of weight-bandwidth-bound time, and it is an ESTIMATE, not MEASURED."),

    ("Analytical stance under test: statistical-power-first - decide the sample size before you look at the delta.",
     "Falsifiable hypothesis H45: the pre-registered sample size is sufficient to detect the minimum interesting effect on both throughput and per-slice quality at the stated power. If the study is underpowered, a null result is uninformative and a positive result is not trustworthy either.",
     "State the minimum interesting effect first - the smallest cost-per-token improvement that would change the deployment decision, and the largest per-slice quality drop that would be tolerated - then compute the sample size that resolves those with the observed variance, using pilot data for the variance estimate. Report confidence intervals on absolute deltas rather than point estimates, and use bootstrap intervals for quality metrics whose sampling distribution is not analytic. Fix the number of eval items and the number of benchmark repetitions before collecting data, and pre-commit the stopping rule, because stopping when the result looks good inflates the apparent effect. Run an A/A comparison of BF16 against itself under the identical harness to establish the noise floor; any claimed delta smaller than the A/A spread is not evidence. Correct for multiple comparisons when reporting many slices, or state that slice-level results are exploratory and require confirmation on a held-out panel. Every reported figure carries MEASURED with its interval and artifact reference, or ESTIMATE with its derivation."),

    ("Analytical stance under test: quality-metric-validity-first - the quality gate must be sensitive to the damage quantization actually does.",
     "Falsifiable hypothesis H46: the chosen quality metric detects a deliberately injected regression of the magnitude the team says it cares about, verified by a positive-control run against a knowingly degraded checkpoint. If the metric cannot see a planted regression, its passing verdict on the real arm carries no information.",
     "Validate the gate before trusting it. Run a positive control: evaluate a checkpoint you know to be worse - a more aggressive bit width, a deliberately mismatched calibration set - and confirm the metric moves beyond its noise band. A metric that passes everything is not a safety property. Prefer metrics tied to the production task over generic aggregate benchmarks, since a benchmark saturated by the model has no headroom to reveal degradation. Report per-slice rather than aggregate, and include the slices most exposed to low-margin token decisions: long-context retrieval, multi-step reasoning, code generation, non-English traffic and structured output. Fix decoding parameters, seeds and stop conditions across arms and record them, because a sampling difference produces quality deltas far larger than the numeric change under test. Retain raw generations so a disputed verdict can be adjudicated from evidence rather than re-run under changed conditions. Rollback gate: any slice breaching its pre-registered indifference threshold blocks adoption even if the aggregate improves."),

    ("Analytical stance under test: rollout-mechanics-first - the qualification result is only as good as the ramp that follows it.",
     "Falsifiable hypothesis H47: at every ramp stage the canary comparison of the quantized path against the concurrently running BF16 path shows no SLO or quality breach beyond the pre-registered abort threshold, on live traffic. If breaches appear only in production, the offline panel was not representative and must be revised before any further ramp.",
     "Design the ramp as a continuing experiment, not a deployment of a settled conclusion. Run both numeric paths concurrently on live traffic with request-level attribution, so comparison happens under identical conditions rather than against yesterday's baseline. Define the stage sequence, the dwell time per stage, the metrics evaluated at each gate, the abort thresholds and the automatic rollback trigger before starting, and record them in the change record. Keep the BF16 path warm and routable throughout so time-to-safe is a routing change rather than a redeploy, and measure that time by rehearsal rather than asserting it. Slice every dashboard and alert by numeric path so an unrelated incident during the ramp can be exonerated quickly. Monitor structured-output compliance and output-length distribution as leading indicators, since they move before aggregate quality scores do. Hold the ramp if any confound appears - a traffic-mix shift, a concurrent engine bump, a checkpoint change - because a ramp with two simultaneous changes yields no attributable result and must be restarted."),

    ("Analytical stance under test: model-and-shape-generalisation-first - state where the conclusion stops applying.",
     "Falsifiable hypothesis H48: the measured benefit holds within the pre-registered boundary of model family, parameter count, bit width, group size, engine version, accelerator generation, context-length range and batch-size range, and re-qualification outside that boundary is mandatory. Testing one point inside and one point outside the boundary either confirms the boundary or shows it was drawn wrongly.",
     "Quantization results are famously non-transferable, and most disputes come from applying a conclusion outside the region where it was measured. Write the boundary explicitly in the result artifact. Mechanisms that break transfer: architectures differ in activation-outlier structure, so a recipe that is benign on one family clips badly on another; mixture-of-experts routing concentrates traffic on a subset of experts whose quantization error is not averaged away; larger models tolerate the same bit width better because redundancy is higher; a newer kernel library may add a fused path that changes the speed result entirely; and a newer accelerator generation may shift the bandwidth-to-compute ratio that made the intervention worthwhile. ESTIMATE: quality damage at fixed bit width tends to decrease with model size and increase with group size; derivation is the redundancy and dynamic-range-per-scale argument only, and it is an ESTIMATE, not MEASURED. Require a re-qualification trigger tied to checkpoint, engine and hardware changes, with an owner, rather than assuming the qualification persists."),

    ("Analytical stance under test: failure-mode-and-incident-first - enumerate how this breaks in production before shipping it.",
     "Falsifiable hypothesis H49: every enumerated failure mode of the quantized path is detectable by an existing alert within the stated detection window, verified by fault injection rather than by inspection. If a failure mode is only detectable by manual investigation, the path is not observable enough to ship.",
     "Enumerate concretely rather than gesturing at risk: silent fallback to a slow reference kernel on unusual shapes, a numerical instability that surfaces only at long context, an accumulator overflow or NaN path that produces empty or truncated outputs, a checkpoint-and-recipe mismatch after a deploy that loads scales for the wrong weights, degraded structured-output compliance breaking downstream parsers, and a driver or kernel-library upgrade silently changing the selected kernel. For each, state the observable signal, the alert that would fire, the expected detection window and the response. Then inject the fault in staging and confirm the alert fires within the window; an untested alert is an assumption. Record numeric path as a trace attribute and a log field so an incident can implicate or exonerate it in minutes. Require a rehearsed rollback with MEASURED time-to-safe and an owner. Rollback gate: any enumerated failure mode without a demonstrated detection path blocks the ramp regardless of the benchmark result."),

    ("Analytical stance under test: decision-record-and-accountability-first - the artifact is the deliverable, not the number.",
     "Falsifiable hypothesis H50: a reviewer with no prior context can, from the decision record alone, identify the claim, the evidence supporting it, the boundary of validity, the residual risks, the rollback trigger and the owner of the post-rollout reconciliation. If any of those is missing, the decision is not auditable and should not be approved.",
     "The durable output of this work is a record that survives the people who produced it. It must contain: the question asked and the decision rule fixed in advance; the pre-registered hypothesis, slices, sample size and stopping rule; the arm manifests showing a bit-width-only symmetric difference; the A/A noise floor; the headline results as MEASURED with intervals and artifact references, with every non-measured figure labelled ESTIMATE with its derivation or ASSUMED with its source; the null-result statement and whether it was reached; the generalisation boundary and re-qualification triggers; the enumerated failure modes with their detection paths; the rehearsed time-to-safe; and a named owner with a date for the post-rollout fleet reconciliation. Prohibit unlabelled numbers anywhere in the document. Publish the record whether the outcome is adoption, rejection or a null, so the next team does not silently repeat the work, and attach an expiry so a stale conclusion is not cited after the engine or hardware has moved. Approval is contingent on the record being complete, not on the result being favourable."),
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
            "rubric never requires a roofline prediction or kernel-level attribution, so a speedup that came from an uncontrolled difference can be reported as a quantization effect",
            "rubric does not separate the weight-bandwidth mechanism from the freed-HBM-to-KV-capacity mechanism, so a memory saving that never became concurrency is credited as cost reduction",
            "no requirement for a representative arrival trace, so a fixed-length synthetic microbenchmark is used for fleet sizing",
            "no statistical power, A/A noise floor or positive-control validation of the quality gate, so an insensitive metric passes a real regression",
            "no generalisation boundary or re-qualification trigger, so a conclusion is carried across model families, engines and hardware generations where it does not hold",
            "no fault-injection-verified detection path, rehearsed time-to-safe or fleet reconciliation obligation, so an unsafe or unrealised change is not excluded",
        ],
        "evidence_required": [
            "per-request raw records: arrival, admission, first-token-to-client, per-token timestamps, output length, terminal status, taken client-side",
            "arm configuration manifests with checkpoint, quantization recipe, calibration-set, engine, kernel-library, driver and container-image hashes showing a bit-width-only symmetric difference",
            "written roofline prediction with derivation, plus kernel-level profile with executed kernel names and achieved bandwidth reconciling measured against predicted speedup",
            "KV block count, memory-utilisation fraction, fragmentation, preemption and swap counts, and sustained concurrency at the SLO for both arms",
            "replayed production arrival trace with input/output length percentiles, prefix-cache hit rates and per-traffic-class results",
            "pre-registered sample size, stopping rule and slices; per-slice quality scores with bootstrap 95% CIs on absolute deltas; BF16-vs-BF16 A/A noise floor; positive-control run against a knowingly degraded checkpoint",
            "structured-output and tool-call compliance panel results with retained raw generations for both arms",
            "independent reproduction on a different host of the same SKU reproducing both arms within the pre-registered noise band",
            "fault-injection records showing each enumerated failure mode fires its alert within the stated detection window, rehearsed rollback with measured time-to-safe, and matched-window fleet GPU-hour, token and replica-count reconciliation",
        ],
        "confidence": 0.62,
    })

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("WROTE", OUT, len(out))
print("IDS", ",".join(r["source_id"] for r in out))
print("OPENINGS_DISTINCT", len({r["corrected_answer"][:200] for r in out}))
