import ast, json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT = f"{EXP}/results/train-batch-0190.jsonl"
START = 1890
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
    ("Analytical stance under test: per-slice-quality-floor-first - an aggregate score cannot clear a quality gate.",
     "Falsifiable hypothesis H61: on every pre-registered traffic slice, the quantized arm's absolute quality delta versus BF16 has a bootstrap 95% confidence interval whose lower bound sits above the pre-registered indifference threshold. If any single slice breaches it, adoption is blocked even when the aggregate improves.",
     "Weight-only quantization damage is concentrated, not uniform, so an aggregate benchmark average is the wrong instrument. Define the slices before measuring, from production traffic rather than from a public suite: long-context requests, non-English inputs, code and structured output, tool-call turns, math and multi-step reasoning, safety-relevant prompts, and the rare-token or domain-jargon slice. Report absolute deltas with bootstrap confidence intervals per slice, not relative percentages, because a percentage on a small base hides a large operational regression. Fix the evaluation set by hash and hold decoding parameters, seeds and prompt templates identical across arms. Require an equal-sized BF16-vs-BF16 A/A run so a reader can distinguish a real slice regression from evaluation noise; without that floor a per-slice claim in either direction is uninterpretable. ESTIMATE: slices whose tokens sit in the low-probability tail are the most exposed, because rounding error changes rank order most where the margin between candidates is smallest; derivation is the margin argument only, and it is an ESTIMATE, not MEASURED. Rollback gate: a slice regression beyond threshold blocks adoption regardless of the cost saving."),

    ("Analytical stance under test: memory-accounting-first - state where the saved bytes actually go before claiming a benefit.",
     "Falsifiable hypothesis H62: the reduction in resident weight bytes measured on device equals the nominal bit-width ratio within a stated tolerance, and the freed bytes become additional KV-cache blocks rather than being reabsorbed by workspace, activation buffers or fragmentation. If freed HBM does not appear as additional usable KV blocks, the capacity claim is void.",
     "Do the byte accounting explicitly rather than assuming the nominal ratio. Resident device memory for a serving replica is weights plus KV cache plus activation and workspace buffers plus allocator fragmentation plus the runtime's own overhead, and weight-only quantization touches only the first term. Measure resident bytes per category on both arms with device-level allocator statistics, not with a formula, since quantized paths often retain BF16 scales, zero-points and dequantization scratch that partially offset the saving. Then verify the freed bytes are usable: report the number of KV blocks the engine actually allocates and the maximum concurrency it admits, because a memory-utilisation fraction that was not raised leaves the saving stranded as idle headroom. Watch fragmentation, which grows when allocation sizes become irregular after quantization and can make a nominal saving unusable. ESTIMATE: scales and zero-points at small group sizes can consume a non-trivial fraction of the nominal saving, and the effect grows as group size shrinks; derivation is bytes-per-group accounting, and it is an ESTIMATE, not MEASURED. Rollback gate: no capacity claim without a measured KV-block delta."),

    ("Analytical stance under test: statistical-power-first - decide what the study can detect before deciding what it found.",
     "Falsifiable hypothesis H63: the pre-computed sample size gives at least the pre-registered power to detect the minimum quality regression the team is unwilling to accept. If the achieved confidence interval is wider than that threshold, the study cannot distinguish 'no regression' from 'undetected regression' and must not be reported as a pass.",
     "The most common false pass in a quantization comparison is an underpowered evaluation reported as equivalence. Compute the required sample size in advance from the pre-registered indifference threshold and the observed per-item variance, per slice rather than in aggregate, since the smallest slices are usually the most exposed and the least powered. Report achieved intervals alongside every verdict so a reader sees what effect sizes were excluded. Distinguish the two claims explicitly: 'we measured no difference' and 'we established equivalence within threshold' are different statements, and only the second justifies adoption. For paired evaluations, use the same items on both arms and analyse the paired differences, which removes item-difficulty variance and materially narrows the interval for the same sample size. Pre-register the stopping rule, because stopping when the numbers look acceptable inflates the false-pass rate by an amount no post-hoc correction can recover. ESTIMATE: paired analysis typically needs a substantially smaller sample than unpaired for the same power, because item variance dominates; derivation is the paired-difference variance argument, and it is an ESTIMATE, not MEASURED."),

    ("Analytical stance under test: warmup-and-steady-state-first - the first minutes of a run are not the deployment.",
     "Falsifiable hypothesis H64: after discarding a pre-registered warmup window, throughput and latency on both arms are stationary across the remainder of the run, and the reported result comes only from the stationary segment. If either arm drifts within the measurement window, the comparison mixes transient and steady-state behaviour.",
     "Serving measurements are contaminated by transients that differ between arms: kernel autotuning and JIT compilation on first execution of each shape, allocator growth until the KV pool reaches its working size, CUDA graph capture, page cache and weight-load effects, and clock behaviour as the device heats and power or thermal limits engage. Any of these can favour one arm simply because it was warmed differently. Define the warmup as a fixed number of requests or a fixed duration, discard it identically on both arms, and then test stationarity on the retained segment rather than assuming it - plot the time series and check for trend, and re-run if a trend is present. Record device clocks, power draw and thermal state throughout, because a sustained-load clock drop is a common cause of a late-run slowdown that gets averaged into the result. Randomise or alternate arm order across repetitions so any residual time-of-day or host-drift effect is not confounded with the arm. ESTIMATE: sustained-load clock reduction can move throughput by a margin comparable to a modest quantization gain; derivation is that power-limited devices reduce frequency under continuous load, and it is an ESTIMATE, not MEASURED."),

    ("Analytical stance under test: numerical-mechanism-first - name the specific error the recipe introduces before predicting where it shows.",
     "Falsifiable hypothesis H65: layers whose weight distributions carry the largest outlier channels show the largest per-layer output error after quantization, and masking those layers back to BF16 recovers most of the measured quality gap. If selective masking does not recover the gap, the damage is not outlier-driven and the chosen mitigation is misdirected.",
     "State the mechanism rather than treating quantization as a black box knob. Weight-only quantization maps each group of weights onto a small integer grid using a scale, so the error per weight is bounded by the group's dynamic range divided by the number of grid levels; a single outlier channel inflates the range for its whole group and degrades every weight sharing that scale. This predicts concrete, testable structure: damage concentrates in layers with heavy-tailed weight distributions, shrinks as group size shrinks, and is worse for symmetric schemes when the distribution is skewed. Test the prediction by measuring per-layer output error against the BF16 reference on a fixed input set, ranking layers, and running a masking arm that keeps the worst layers in BF16 while quantizing the rest. This both localises the damage and gives a cheap mitigation with a measurable cost in bytes. Record group size, symmetry, and any outlier-handling or rotation step as part of the recipe identity, since these change the mechanism entirely. ESTIMATE: error per weight scales roughly with group dynamic range over grid levels, so halving group size reduces error sublinearly with a bytes cost that grows linearly in scales; derivation is the grid-spacing argument, and it is an ESTIMATE, not MEASURED."),

    ("Analytical stance under test: operational-reversibility-first - the deployment plan is part of the comparison.",
     "Falsifiable hypothesis H66: a rehearsed rollback from the quantized arm to BF16 completes within the pre-registered time budget without exceeding error-budget burn, on the actual production control plane. If rollback has not been rehearsed end to end, the adoption decision rests on an untested assumption.",
     "A favourable benchmark does not make a change safe to ship, because the risk is concentrated in the transition rather than the steady state. Specify the rollout as a canary with explicit traffic fractions, dwell time at each step, and automated abort conditions tied to the same tail-latency, per-slice quality and structured-output conformance metrics used in qualification, evaluated continuously rather than reviewed after the fact. Require that the BF16 checkpoint remain resident and loadable for the whole rollout, and rehearse the rollback under load to measure how long weight reload, engine restart and warmup actually take, since that duration is the real exposure window and is routinely underestimated. Name the on-call owner, the dashboards, the abort authority and the maximum acceptable error-budget burn before the first percent of traffic moves. Handle mixed-fleet state explicitly: during rollout both arms serve simultaneously, so any client that caches responses or compares outputs across replicas will see inconsistency, and that must be either tolerated by design or prevented by routing. ESTIMATE: reload plus warmup dominates rollback time and grows with weight size and replica count; derivation is that rollback is bounded by the slowest replica's load-and-warm path, and it is an ESTIMATE, not MEASURED. Rollback gate: no canary starts before a timed rollback rehearsal exists."),

    ("Analytical stance under test: batch-composition-first - the request mix is an experimental variable, not background.",
     "Falsifiable hypothesis H67: replaying the arms against the production request-mix trace reproduces the ranking obtained under the synthetic fixed-length workload. If the ranking inverts or the effect size collapses under the real mix, the synthetic result does not describe the deployment.",
     "Fixed-length synthetic workloads systematically misrepresent continuous-batching engines, because the achievable batch size, the prefill-to-decode ratio and the scheduler's preemption behaviour all depend on the distribution of input and output lengths, not on their means. Weight-only quantization acts mainly on the decode phase, so a workload with unusually long outputs flatters it and a prefill-heavy workload hides it. Replay a captured production trace with its real arrival process, input-length and output-length distributions, cancellation rate and burst structure, and compare against the synthetic result rather than substituting one for the other. Record the achieved batch-size distribution, the prefill-decode time split and preemption counts on both arms, because these explain any divergence and turn a surprising result into a diagnosis. Preserve arrival timing rather than replaying as fast as possible, since an open-loop trace replayed closed-loop removes queueing, which is exactly where tail latency lives. ESTIMATE: the measured decode-side gain scales roughly with the decode share of total step time under the given mix; derivation is time-share weighting, and it is an ESTIMATE, not MEASURED. Rollback gate: a claim supported only by synthetic workload is scoped to that workload in writing."),

    ("Analytical stance under test: alternatives-and-opportunity-cost-first - compare the lever against its substitutes, not only against doing nothing.",
     "Falsifiable hypothesis H68: within the same engineering budget, weight-only quantization delivers a larger cost-per-token reduction at equal quality than the leading alternatives measured on the same harness - a smaller checkpoint, KV-cache quantization, better batching and admission policy, prefix caching, or speculative decoding. If any alternative matches it at lower quality risk, the alternative wins.",
     "The stated goal is lower serving cost, and quantization is one of several levers, so a comparison against BF16 alone answers the wrong question. Enumerate the candidate levers and measure them on the same harness, same traffic and same SLO: a smaller or distilled checkpoint, KV-cache quantization or grouped-query attention, prefix and prompt caching, improved scheduling and admission control, speculative decoding, and simple right-sizing of replica count and memory-utilisation fraction. Several of these are orthogonal to quantization and stack with it, so also record which combinations were tested and which are assumed additive, since assumed additivity is a frequent source of overstated projections. Include the engineering and maintenance cost of each option, not only its runtime effect: a recipe that requires per-model calibration, per-shape kernel attestation and re-qualification on every driver bump carries ongoing cost that a scheduler change does not. Report the ranking with quality risk attached to each entry so the decision is made on the full trade. ESTIMATE: at long context the KV-side levers usually dominate weight-side ones, following the byte-accounting argument above, and it is an ESTIMATE, not MEASURED."),

    ("Analytical stance under test: reproducibility-and-adjudication-first - a result nobody else can rerun is not evidence.",
     "Falsifiable hypothesis H69: an independent engineer, given only the published artefact bundle, reproduces the headline numbers within the pre-registered reproduction band on the same hardware class. If reproduction fails, the original result is treated as provisional and not used for an adoption decision.",
     "Make the artefact bundle the deliverable rather than the summary slide. It must contain the harness commit, the exact command lines, both arm manifests with all hashes, the raw per-request records, the raw generations for every quality evaluation, the calibration-set hash, the kernel profiles, the analysis scripts, and the pre-registered decision rule with a timestamp that predates data collection. Fix and record all seeds, and state which parts of the pipeline remain nondeterministic - kernel reduction order, batching-dependent numerics, and scheduler nondeterminism under load - because claiming bit-exact reproducibility where it does not exist destroys trust in the parts that are reproducible. Nominate a reproduction owner who did not run the original study and give them a deadline; without a named owner reproduction is never attempted. Define the reproduction band in advance so a small difference is not litigated after the fact. ESTIMATE: run-to-run variation on a loaded shared cluster is often the same order as a modest quantization effect, which is why an A/A noise floor and a stated band are prerequisites rather than niceties, and it is an ESTIMATE, not MEASURED."),

    ("Analytical stance under test: security-licence-and-supply-chain-first - the artefact's provenance is an operational property.",
     "Falsifiable hypothesis H70: the quantized checkpoint is byte-reproducible from the recorded base checkpoint, recipe, calibration-set hash and toolchain version by an independent rebuild. If the deployed artefact cannot be rebuilt from recorded inputs, its provenance is unverified and it should not be admitted to production regardless of measured performance.",
     "A quantized checkpoint is a new artefact with its own supply chain, and the comparison is incomplete if it ignores that. Record the base checkpoint hash, its licence and redistribution terms, the quantization toolchain and version, the calibration-set contents and hash including whether it contains customer or otherwise restricted data, and the resulting artefact hash and signature. Require an independent rebuild from those recorded inputs, since a rebuild that does not match indicates undocumented manual steps that will not survive the next re-qualification. Treat the checkpoint loader as an attack surface and prefer formats that do not execute code on load, scanning artefacts before admission. Note the data-governance exposure specifically: calibration on production traffic can embed customer data characteristics into distributed weights, which may violate retention or residency commitments, and that question must be answered before the artefact leaves the build system rather than after deployment. Assign storage, retention and re-qualification ownership for the artefact, because an unowned checkpoint outlives the study that produced it. Rollback gate: an artefact that fails independent rebuild or admission scanning is not deployed."),
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
            "rubric asks for aggregate quality comparison without a per-slice floor or A/A noise baseline, so a concentrated regression on long-context, non-English, code or tool-call traffic passes unnoticed",
            "no memory byte accounting, so a nominal weight-byte saving is claimed as capacity even when scales, workspace and fragmentation reabsorb it and KV blocks do not increase",
            "no statistical power or stopping rule, so an underpowered null is reported as equivalence",
            "no warmup and stationarity discipline, so autotuning, allocator growth and sustained-load clock drop are averaged into the arm comparison",
            "no request-mix realism requirement, so a synthetic fixed-length decode-heavy workload is projected onto production traffic",
            "no rollback rehearsal, artefact-provenance, licence or calibration-data-governance requirement, so the transition and supply-chain risks are entirely outside the stated comparison",
        ],
        "evidence_required": [
            "per-slice absolute quality deltas with bootstrap 95% CIs on production-derived slices, plus an equal-sized BF16-vs-BF16 A/A noise floor and retained raw generations",
            "device allocator statistics per category (weights, KV, activation/workspace, fragmentation) on both arms, with measured KV-block count and admitted-concurrency delta",
            "pre-registered indifference threshold, power calculation, sample size and stopping rule timestamped before data collection, with paired-difference analysis",
            "time-series throughput and latency with the discarded warmup window marked, stationarity check, and device clock/power/thermal traces for both arms",
            "per-layer output-error ranking against the BF16 reference plus a selective-BF16-masking arm; recipe identity recording group size, symmetry and any outlier or rotation step",
            "production trace replay preserving arrival timing, input/output length distributions and cancellations, with achieved batch-size distribution, prefill-decode split and preemption counts",
            "same-harness measurements of alternative levers (smaller checkpoint, KV quantization, prefix caching, scheduling, speculative decoding) with engineering and maintenance cost attached",
            "timed rollback rehearsal under load with canary fractions, dwell times, automated abort conditions, named owner and error-budget limit",
            "artefact bundle enabling independent reproduction, plus independent byte-reproducible rebuild of the quantized checkpoint from base checkpoint, recipe, calibration hash and toolchain version, with licence and calibration-data governance review",
        ],
        "confidence": 0.62,
    })

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("WROTE", OUT, len(out))
print("IDS", ",".join(r["source_id"] for r in out))
print("OPENINGS_DISTINCT", len({r["corrected_answer"][:200] for r in out}))
