"""Quantisation and numerics mechanisms (topic: quantization)."""
from __future__ import annotations

from core import Mechanism, Quant, Setting, fmt_int, gib, register


def q_weight_bytes_saved(s: Setting) -> Quant:
    cur = s.weight_bytes_per_gpu
    int8 = int(s.params_b * 1e9) // max(s.tp, 1)
    int4 = int8 // 2
    return Quant(
        label="the per-device weight bytes at bf16, int8 and int4 for this shard",
        steps=[
            f"Current at {s.dtype}: {s.params_b}e9 * {s.dtype_bytes} / TP{s.tp} = {gib(cur)}",
            f"At int8 (1 B/param): {s.params_b}e9 / {s.tp} = {gib(int8)}",
            f"At int4 (0.5 B/param): {gib(int4)}",
            f"Decode re-reads the full shard every step, so bytes per step fall in the same ratio",
        ],
        value=f"{gib(cur)} now, {gib(int8)} at int8, {gib(int4)} at int4 per device",
        interpretation=(
            "The saving is a bandwidth saving during decode, where weights are re-read every step. "
            "During prefill the phase is compute-bound and the same reduction buys very little."),
    )


def q_kv_quant(s: Setting) -> Quant:
    per_tok = s.kv_bytes_per_token
    fp8 = per_tok // 2 if s.dtype_bytes == 2 else per_tok
    budget = s.kv_budget_bytes
    return Quant(
        label="the concurrency bought by halving cache precision",
        steps=[
            f"KV per token at {s.dtype}: {fmt_int(per_tok)} B",
            f"At 8-bit cache: {fmt_int(fp8)} B per token",
            f"Full-context request: {gib(per_tok * s.ctx)} becomes {gib(fp8 * s.ctx)}",
            f"Seats in a {gib(budget)} budget: {budget // max(per_tok * s.ctx, 1)} becomes "
            f"{budget // max(fp8 * s.ctx, 1)}",
        ],
        value=(f"seats rise from {budget // max(per_tok * s.ctx, 1)} to "
               f"{budget // max(fp8 * s.ctx, 1)} at full context"),
        interpretation=(
            "The capacity gain is exact arithmetic. The quality cost is not, and it cannot be inferred "
            "from the memory saving; it has to be measured on the served task."),
    )


def q_outlier_range(s: Setting) -> Quant:
    return Quant(
        label="why one scale per tensor fails when activations contain outliers",
        steps=[
            f"A per-tensor scale must cover the largest magnitude in the tensor",
            f"Activation outliers in transformer hidden states routinely exceed the typical magnitude "
            f"by two orders of magnitude",
            f"With 8 bits and a scale set by the outlier, ordinary values occupy the lowest few levels",
            f"Effective resolution for the bulk of the {fmt_int(s.hidden)} hidden dimensions collapses "
            f"to a small number of distinct levels",
        ],
        value="a scale set by outliers leaves the bulk of values with very few usable levels",
        interpretation=(
            "This is why per-channel or group-wise scaling exists. It is not a refinement; without it "
            "the quantised representation carries far less information than its bit width suggests."),
    )


def q_calibration_size(s: Setting) -> Quant:
    return Quant(
        label="the coverage a calibration set must have to set scales safely",
        steps=[
            f"Scales are derived from observed activation ranges on the calibration set",
            f"Any range not represented in calibration is clipped at serving time",
            f"Context lengths up to {fmt_int(s.ctx)} must appear, since positional statistics differ "
            f"with length",
            f"Every serving task type must appear, since ranges are input-dependent",
        ],
        value=f"calibration must span the served length range up to {fmt_int(s.ctx)} and every task type",
        interpretation=(
            "Calibration set composition, not size, is what determines whether serving inputs fall "
            "inside the calibrated range. A large sample drawn from one task is worse than a small "
            "stratified one."),
    )


def q_kernel_gap(s: Setting) -> Quant:
    return Quant(
        label="the arithmetic that a missing fused kernel gives back",
        steps=[
            f"With a fused kernel, weights are read at reduced precision and used directly",
            f"Without one, weights are read at reduced precision and expanded to {s.dtype} before use",
            f"The expansion writes and re-reads a full-precision copy, restoring the original traffic",
            f"Per-device shard is {gib(s.weight_bytes_per_gpu)}, so the round trip is of that order",
        ],
        value=f"an unfused path can restore traffic of roughly {gib(s.weight_bytes_per_gpu)} per step",
        interpretation=(
            "The stored format and the executed format are different questions. Memory footprint "
            "falls either way; step time only falls when the kernel consumes the compressed form "
            "directly."),
    )


def q_reduction_order(s: Setting) -> Quant:
    return Quant(
        label="how many partial sums a reduction order can permute at this parallel degree",
        steps=[
            f"Each layer reduces across TP{s.tp} ranks",
            f"Floating-point addition is not associative, so a different arrival order gives a "
            f"different sum in the last bits",
            f"Across {s.layers} layers the difference compounds through the residual stream",
            f"At a sampling boundary the compounded difference can change the selected token",
        ],
        value=f"{s.tp}-way reductions across {s.layers} layers, each order-sensitive in the last bits",
        interpretation=(
            "Bitwise reproducibility across runs is not guaranteed by fixing the seed alone. Any "
            "comparison that requires identical outputs must also fix the reduction order or accept "
            "divergence as expected."),
    )


def q_eval_power(s: Setting) -> Quant:
    n = 100
    return Quant(
        label="what a 100-item evaluation can and cannot detect",
        steps=[
            f"With {n} items, a per-item pass rate has a standard error of at most "
            f"0.5 / sqrt({n}) = {0.5 / (n ** 0.5):.3f}",
            f"A two-arm comparison roughly doubles that, so differences below about "
            f"{2 * 0.5 / (n ** 0.5) * 100:.0f} percentage points are inside noise",
            f"Quantisation regressions frequently sit below that band",
            f"Detecting a 2-point difference needs on the order of {int((0.5 / 0.01) ** 2)} items",
        ],
        value=f"a {n}-item set cannot resolve differences below roughly "
              f"{2 * 0.5 / (n ** 0.5) * 100:.0f} percentage points",
        interpretation=(
            "An evaluation that reports no regression is only informative if it could have detected "
            "one. The detectable band must be stated with the result."),
    )


def q_fp8_range(s: Setting) -> Quant:
    return Quant(
        label="the dynamic range an 8-bit float leaves after scaling",
        steps=[
            "An 8-bit float trades mantissa bits for exponent range against int8's uniform spacing",
            "Values outside the representable range saturate rather than wrapping",
            "A scale factor recentres the distribution but cannot widen the range",
            f"Both tails matter: activations across {fmt_int(s.hidden)} channels are not symmetric",
        ],
        value="scaling recentres the distribution; it does not extend the representable range",
        interpretation=(
            "Saturation is silent. It produces plausible outputs with a systematic bias, so it is "
            "found by measuring the served task rather than by watching for errors."),
    )


def q_regime(s: Setting) -> Quant:
    wpg = s.weight_bytes_per_gpu
    per_tok = s.kv_bytes_per_token
    kv_half = per_tok * (s.ctx // 2)
    b = s.concurrency
    return Quant(
        label="the share of decode traffic that weight quantisation can actually reduce",
        steps=[
            f"Weight traffic per step (fixed): {gib(wpg)}",
            f"KV traffic per step at batch {b}, half context: {b} * {gib(kv_half)} = {gib(kv_half * b)}",
            f"Weight share = {gib(wpg)} / ({gib(wpg)} + {gib(kv_half * b)}) = "
            f"{wpg / max(wpg + kv_half * b, 1) * 100:.1f}%",
            f"Halving weight bytes reduces total traffic by at most half of that share",
        ],
        value=f"weights are {wpg / max(wpg + kv_half * b, 1) * 100:.1f}% of decode traffic at batch {b}",
        interpretation=(
            "The upper bound on speedup is set by the weight share, which falls as batch and context "
            "grow. At high concurrency the cache term dominates and weight quantisation buys little."),
    )


def q_dequant_overhead(s: Setting) -> Quant:
    return Quant(
        label="why the quantisation benefit disappears at small batch",
        steps=[
            "Dequantisation and scale application cost arithmetic proportional to the weight count",
            f"That cost is paid per step regardless of how many tokens the step produces",
            f"At batch 1 it is charged against a single token; at batch {s.concurrency} it is "
            f"amortised across {s.concurrency}",
            "So the net benefit rises with batch size even though the memory saving does not",
        ],
        value=f"dequantisation cost is amortised across the batch, from 1 to {s.concurrency} tokens",
        interpretation=(
            "A quantised deployment benchmarked at batch 1 can look slower than the unquantised one "
            "while being faster in production, and the reverse is equally possible. The batch size "
            "must match the serving regime."),
    )


register(
    Mechanism(
        key="weight_quant_regime", topic="quantization",
        title="weight quantisation helps the memory-bound decode phase and barely helps prefill",
        concepts=("quantization", "roofline", "prefill"),
        symptom="Weight quantisation delivered a clear decode speedup and left time to first token essentially unchanged.",
        chain="Decode re-reads the whole weight shard for every step and is limited by bytes moved, so halving weight bytes halves that term; prefill is limited by arithmetic over many positions, and the same reduction does not change the arithmetic.",
        metric="Speedup measured separately for time to first token and for time per output token.",
        signature="Per-output-token time improves roughly in proportion to the weight-byte reduction while first-token time is flat.",
        confounders=(
            "A kernel change bundled with the quantisation, which alters compute time independently.",
            "Batch size changing because more memory became available, which moves both phases.",
            "Prefix cache hits reducing prefill work, which mimics a first-token improvement.",
        ),
        fixes=(
            "Report the two phases separately before crediting quantisation with any end-to-end number.",
            "Set expectations from the measured weight share of decode traffic rather than from the bit-width ratio.",
            "Quantise the cache as well if the objective is concurrency rather than per-token latency.",
        ),
        rollback="Return to the previous precision if output quality regresses on the served task, regardless of the latency gain, since quality is the constraint that quantisation trades against.",
        options=("reporting the two phases separately", "quantising the cache as well as the weights"),
        tradeoff="whether the objective is per-token latency or admitted concurrency",
        flip="concurrency rises to the point where cache traffic dominates, at which point weight precision stops being the lever",
        falsifier="first-token time improves in proportion to the weight-byte reduction",
        wrong_claim="We halved the weight bytes, so the model should run about twice as fast.",
        wrong_why="Only the weight portion of decode traffic is halved, and prefill is not affected at all, so the achievable speedup is bounded by the weight share of the phase in question.",
        threshold="Expect decode speedup no greater than the measured weight share of decode traffic times the byte reduction.",
        cost="Quality risk is taken across the whole service while the latency benefit accrues only to the decode phase.",
        scaling="The weight share of traffic falls as batch and context grow, so the benefit shrinks exactly as the service scales up.",
        quant=q_regime,
    ),
    Mechanism(
        key="kv_quant_quality_cost", topic="quantization",
        title="cache quantisation buys concurrency in exact arithmetic and costs quality in unmeasured amounts",
        concepts=("kv_cache", "quantization", "evaluation"),
        symptom="Cache quantisation raised admitted concurrency as predicted, and nobody can say what it cost in answer quality.",
        chain="Cache bytes per token fall exactly with precision, so the capacity gain is arithmetic; the quality effect depends on how attention uses the degraded values and is only observable on the served task.",
        metric="Task-level quality on the served workload, measured at the same time as the capacity gain.",
        signature="Capacity matches the arithmetic prediction exactly while quality changes are small, task-dependent and invisible in serving metrics.",
        confounders=(
            "Batch size changing with the new capacity, which alters outputs through scheduling rather than precision.",
            "Longer contexts being admitted after the change, which shifts the input distribution.",
            "Sampling temperature masking small quality differences behind run-to-run variation.",
        ),
        fixes=(
            "Measure task quality on a set large enough to detect a difference worth caring about before enabling it broadly.",
            "Quantise only the older portion of the cache where attention weight is lowest, if the runtime supports it.",
            "Return to full-precision cache and buy capacity with devices instead, if quality cannot be established.",
        ),
        rollback="Revert to full-precision cache if the quality measurement is inconclusive rather than treating an inconclusive result as a pass.",
        options=("measuring task quality before broad enablement", "quantising only the older cache entries"),
        tradeoff="whether a quality difference large enough to matter can be detected on the available evaluation",
        flip="the evaluation turns out to lack the power to detect the difference that matters, at which point partial quantisation limits exposure rather than resolving it",
        falsifier="a properly powered quality evaluation shows no difference on the served task",
        wrong_claim="Serving metrics are unchanged after enabling cache quantisation, so it is safe.",
        wrong_why="Serving metrics measure latency and throughput, which the change improves; answer quality is not among them, so unchanged serving metrics carry no information about the risk being taken.",
        threshold="Require a quality evaluation powered to detect the smallest regression worth reverting for, and state that band with the result.",
        cost="Capacity bought with unmeasured quality loss is capacity whose real price is unknown at the time of purchase.",
        scaling="Exposure grows with the share of traffic served under the quantised cache, so a partial rollout bounds risk in proportion to its coverage.",
        quant=q_kv_quant,
    ),
    Mechanism(
        key="activation_outliers", topic="quantization",
        title="activation outliers make a single per-tensor scale destroy ordinary values",
        concepts=("quantization", "outliers", "scaling"),
        symptom="Post-training quantisation that passed a perplexity check produces visibly degraded answers on real prompts.",
        chain="A per-tensor scale must be large enough to represent the largest activation, and transformer hidden states contain rare values far above the typical magnitude, so the ordinary values are compressed into a few quantisation levels.",
        metric="Activation magnitude distribution per channel, reported with its upper tail rather than as a mean.",
        signature="A small number of channels carry magnitudes far above the rest, and quantisation error concentrates in the remaining channels rather than in the outlier ones.",
        confounders=(
            "A calibration set that never triggered the outlier channels, which hides the problem rather than removing it.",
            "Weight quantisation error applied at the same time, which contributes to the same output degradation.",
            "Aggregate perplexity averaging away failures concentrated on specific inputs.",
        ),
        fixes=(
            "Move to per-channel or group-wise scaling so outlier channels do not set the scale for the rest.",
            "Keep the identified outlier channels at higher precision and quantise the remainder.",
            "Apply a smoothing transform that shifts magnitude from activations into weights before quantising.",
        ),
        rollback="Return to the previous precision if per-channel scaling does not recover the quality gap, since that indicates the loss is not from scale coverage.",
        options=("moving to per-channel or group-wise scaling", "keeping outlier channels at higher precision"),
        tradeoff="whether the outliers are confined to a stable, identifiable set of channels",
        flip="outlier channels move with the input distribution, at which point a fixed high-precision set stops covering them and the scaling granularity has to carry it",
        falsifier="activation magnitudes are uniform across channels and quantisation error is evenly distributed",
        wrong_claim="Perplexity moved by less than a point after quantisation, so the model is essentially unchanged.",
        wrong_why="Aggregate perplexity averages over tokens and hides degradation concentrated on the inputs that trigger the outlier channels, which is where the served workload actually lives.",
        threshold="Require per-channel magnitude statistics before choosing a scaling granularity, and reject per-tensor scaling where the channel maxima span orders of magnitude.",
        cost="A quantised deployment that must be reverted after release costs the migration twice and spends the intervening quality on users.",
        scaling="Outlier magnitude tends to grow with model size, so a recipe validated on a smaller model does not transfer upward.",
        quant=q_outlier_range,
    ),
    Mechanism(
        key="calibration_mismatch", topic="quantization",
        title="quantisation scales inherit the calibration set's distribution, not the serving distribution",
        concepts=("quantization", "calibration", "distribution_shift"),
        symptom="A quantised model performs well on the evaluation used during calibration and poorly on a specific class of production traffic.",
        chain="Scales are fitted to activation ranges observed during calibration, so any serving input that produces ranges outside those observed is clipped, and the clipping is systematic rather than random.",
        metric="Fraction of serving activations falling outside the calibrated range, sampled by task type and context length.",
        signature="Clipping concentrates in the task types absent from calibration and in context lengths beyond those calibrated.",
        confounders=(
            "A genuinely harder task class, which would degrade at full precision as well.",
            "Prompt template differences between calibration and serving, which change ranges without changing task difficulty.",
            "Sequence length effects on positional statistics, which shift ranges independently of content.",
        ),
        fixes=(
            "Stratify the calibration set across every served task type and across the served length range.",
            "Measure the out-of-range fraction on production traffic before deploying the quantised artifact.",
            "Recalibrate whenever the prompt template or the served task mix changes materially.",
        ),
        rollback="Revert to the previous artifact if the out-of-range fraction on production traffic exceeds what was seen during calibration, rather than waiting for a quality complaint.",
        options=("stratifying the calibration set across tasks and lengths", "measuring the out-of-range fraction on production traffic"),
        tradeoff="whether the serving distribution is stable and known well enough to be sampled in advance",
        flip="the served task mix changes after calibration, at which point a stratified historical set no longer covers the traffic and live measurement becomes the only control",
        falsifier="the out-of-range fraction is negligible on the failing traffic class",
        wrong_claim="The calibration set had thousands of samples, so the ranges are well estimated.",
        wrong_why="Coverage rather than count determines whether serving ranges were observed, and thousands of samples from one task type leave every other task type uncalibrated.",
        threshold="Require the calibration set to cover every served task type and the full served length range before the artifact is deployable.",
        cost="Recalibration is cheap; discovering the gap through degraded production answers is not.",
        scaling="Risk grows with the diversity of served traffic, so a broadening product surface silently invalidates an older calibration.",
        quant=q_calibration_size,
    ),
    Mechanism(
        key="quant_kernel_gap", topic="quantization",
        title="a compressed storage format without a matching kernel gives memory back and speed nowhere",
        concepts=("quantization", "kernels", "memory_bandwidth"),
        symptom="Weights are stored in a reduced-precision format, device memory fell as expected, and step time did not improve at all.",
        chain="If no kernel consumes the compressed format directly, the runtime expands weights to full precision before the matrix multiply, so the traffic saved on the read is spent again on the expansion and the compute path is unchanged.",
        metric="Bytes read per decode step and the presence of a dequantisation kernel in the step profile.",
        signature="A dequantisation kernel appears in the profile and total bytes moved per step are close to the unquantised baseline.",
        confounders=(
            "Falling back to a generic kernel for an unsupported shape, which is slow for a different reason.",
            "The measurement running at a batch size where dequantisation cost is not amortised.",
            "Memory savings enabling a larger batch, which changes step time independently of the format.",
        ),
        fixes=(
            "Confirm from the step profile that a fused kernel for the chosen format exists on this hardware before adopting it.",
            "Choose a format the installed kernel library supports natively rather than the smallest available one.",
            "Keep the previous precision if no fused path exists, since the memory saving alone rarely justifies the quality risk.",
        ),
        rollback="Revert the format if the step profile shows a dequantisation kernel, since that confirms the compressed form is not being consumed directly.",
        options=("confirming a fused kernel exists before adopting a format", "choosing a natively supported format"),
        tradeoff="whether the installed kernel library consumes the chosen format directly on this hardware",
        flip="a fused kernel appears in a later library version, at which point the format choice can be revisited on speed rather than support grounds",
        falsifier="no dequantisation kernel appears in the profile and bytes per step fall in line with the format",
        wrong_claim="The weights are int4 on disk and in memory, so the model is reading a quarter of the bytes.",
        wrong_why="Storage format and execution format are separate; without a fused kernel the runtime expands to full precision before use and the read saving is cancelled by the expansion.",
        threshold="Require the step profile to show no dequantisation kernel before a quantised format is treated as a speed improvement.",
        cost="Quality risk is taken for a memory saving alone, while the latency benefit that justified it never materialises.",
        scaling="The expansion cost scales with weight count, so larger models pay more for the missing kernel rather than less.",
        quant=q_kernel_gap,
    ),
    Mechanism(
        key="reduction_nondeterminism", topic="quantization",
        title="parallel reduction order makes outputs vary in the last bits even at temperature zero",
        concepts=("determinism", "collectives", "numerics"),
        symptom="Two runs with an identical seed, prompt and model produce outputs that diverge after some number of tokens.",
        chain="Floating-point addition is not associative, so a reduction whose operands arrive in a different order yields a slightly different sum, and across many layers the difference can cross a sampling boundary and change the selected token.",
        metric="Position of first divergence between two identical runs, together with the logit gap at that position.",
        signature="Divergence begins at a position where the top two logits are nearly equal, and the prefix before it is bitwise identical.",
        confounders=(
            "A genuinely non-zero sampling temperature, which produces divergence for an ordinary reason.",
            "Different kernel selection between runs from autotuning, which changes arithmetic rather than order.",
            "Batch composition differing between runs, which changes the shapes involved and therefore the reduction.",
        ),
        fixes=(
            "Fix batch composition and disable autotuning when bitwise reproducibility is required.",
            "Select deterministic collective and kernel implementations where the framework offers them, accepting their cost.",
            "Change the acceptance criterion to a distributional comparison rather than bitwise equality, where determinism is not worth its price.",
        ),
        rollback="Return to the faster non-deterministic path once the comparison that required determinism is complete, since the deterministic path costs throughput.",
        options=("fixing batch composition and disabling autotuning", "selecting deterministic collective implementations"),
        tradeoff="whether bitwise reproducibility is worth the throughput it costs for this purpose",
        flip="the deterministic path's throughput cost makes the experiment impractical at the needed sample size, at which point a distributional criterion is the better choice",
        falsifier="the two runs are bitwise identical throughout, which removes reduction order as the explanation",
        wrong_claim="Temperature is zero and the seed is fixed, so any output difference must be a bug.",
        wrong_why="Greedy decoding removes sampling randomness but not arithmetic non-associativity, and near-tied logits convert a last-bit difference into a different token.",
        threshold="Treat divergence at positions with near-tied logits as expected; treat divergence at positions with a wide logit gap as a genuine defect.",
        cost="Chasing a reproducibility bug that is arithmetic in origin consumes engineering time without any defect to find.",
        scaling="Divergence probability grows with parallel degree, layer count and output length, so larger deployments diverge sooner.",
        quant=q_reduction_order,
    ),
    Mechanism(
        key="underpowered_quality_eval", topic="quantization",
        title="an evaluation that could not detect a regression is not evidence there was none",
        concepts=("evaluation", "statistical_power", "quantization"),
        symptom="A hundred-item evaluation showed no quality difference after a precision change, and the change was approved on that basis.",
        chain="The smallest difference a comparison can resolve is set by the sample size and the metric's variation, and a hundred items cannot resolve the few-percentage-point regressions that precision changes typically produce.",
        metric="The minimum detectable difference of the evaluation, computed from its sample size and observed variation, reported alongside the result.",
        signature="The observed difference is smaller than the evaluation's own detectable band, so the result is uninformative rather than negative.",
        confounders=(
            "A truncation or harness defect suppressing scores in both arms, which hides a real difference.",
            "Item difficulty being concentrated, so most items are passed or failed by both arms and carry no signal.",
            "Repeated evaluation on the same set, which converts it into a target rather than a measurement.",
        ),
        fixes=(
            "Report the minimum detectable difference with every evaluation result.",
            "Size the evaluation from the smallest regression worth reverting for, before running it.",
            "Use a paired comparison on identical inputs, which removes item difficulty from the variance and needs fewer items.",
        ),
        rollback="Treat an underpowered result as a reason to keep the previous artifact rather than as clearance to proceed.",
        options=("reporting the minimum detectable difference", "using a paired comparison on identical inputs"),
        tradeoff="whether the evaluation can resolve the smallest regression that would change the decision",
        flip="the items are too easy or too hard for a paired design to separate the arms, at which point only a larger and harder set will do",
        falsifier="the observed difference is larger than the computed detectable band",
        wrong_claim="The evaluation showed no significant difference, so quality is preserved.",
        wrong_why="Absence of a detected difference is not evidence of absence when the design could not have detected the difference that matters, and that band was never computed.",
        threshold="Require the stated minimum detectable difference to be smaller than the regression that would trigger a rollback.",
        cost="Accelerator hours spent on an underpowered evaluation buy a number that cannot support the decision it is used for.",
        scaling="Required sample size grows as the square of the resolution demanded, so halving the detectable band costs four times the evaluation.",
        quant=q_eval_power,
    ),
    Mechanism(
        key="fp8_saturation", topic="quantization",
        title="eight-bit float saturates silently at the tails and scaling cannot widen the range",
        concepts=("fp8", "dynamic_range", "numerics"),
        symptom="An eight-bit deployment produces fluent output with a consistent bias on a subset of inputs and no errors anywhere.",
        chain="Values beyond the representable range saturate to the extreme rather than raising, so the affected activations are systematically pulled toward the boundary, and a scale factor recentres the distribution without extending what it can represent.",
        metric="Saturation rate per tensor, counted at both tails and reported by input class.",
        signature="Saturation concentrates on a specific input class and at one tail rather than being spread evenly.",
        confounders=(
            "Clipping from the calibration range rather than from the format's own limit, which has a different fix.",
            "Outlier channels, which cause resolution loss rather than saturation.",
            "Accumulation in reduced precision, which introduces error without any value reaching the boundary.",
        ),
        fixes=(
            "Instrument saturation counters per tensor and treat a non-zero rate as a finding rather than as noise.",
            "Re-fit scales against the observed serving distribution rather than the calibration one.",
            "Keep the affected tensors at higher precision, since no scale choice makes the range wider.",
        ),
        rollback="Revert the affected tensors to higher precision if the saturation rate does not fall to zero after rescaling, because rescaling cannot fix a range problem.",
        options=("instrumenting saturation counters per tensor", "keeping the affected tensors at higher precision"),
        tradeoff="whether the values exceeding the range are rare enough to lose or are carrying the signal",
        flip="the saturating values turn out to carry the information the task depends on, at which point no rate is acceptable and precision must rise",
        falsifier="saturation counters are zero across all tensors on the affected input class",
        wrong_claim="No numerical errors or NaNs are being reported, so the reduced precision is operating correctly.",
        wrong_why="Saturation is defined behaviour rather than an error, so it produces no diagnostic; it appears only as a systematic bias in the output.",
        threshold="Treat any non-zero saturation rate on a serving tensor as requiring investigation before the artifact is promoted.",
        cost="A systematic bias shipped to production is more expensive to detect and to unwind than the accelerator hours a higher precision would have cost.",
        scaling="Saturation rate rises with sequence length and with model size as activation magnitudes grow, so it worsens on exactly the long-context traffic that motivated the memory saving.",
        quant=q_fp8_range,
    ),
    Mechanism(
        key="dequant_batch_amortisation", topic="quantization",
        title="dequantisation overhead is amortised by batch, so the measured benefit depends on the batch it was measured at",
        concepts=("quantization", "batching", "benchmarking"),
        symptom="A quantised build benchmarked slower than the baseline in a single-stream test and faster in production.",
        chain="Scale application and format conversion cost arithmetic proportional to the weight count and are paid once per step regardless of batch size, so at batch one they are charged against a single token and at production batch they are spread across many.",
        metric="Step time and tokens produced per step, at several batch sizes for both builds.",
        signature="The crossover batch size where the quantised build overtakes the baseline is visible in the sweep and is above one.",
        confounders=(
            "Memory savings permitting a larger batch, which is a real benefit but a different one.",
            "Kernel selection differing by batch size, which moves both builds unevenly.",
            "Warm-up and capture costs contaminating the smallest batch measurements.",
        ),
        fixes=(
            "Benchmark at the production batch size rather than at batch one.",
            "Report the crossover batch size with any quantisation speedup claim.",
            "Choose the format from the sweep rather than from a single operating point.",
        ),
        rollback="Return to the previous build if the production batch size sits below the measured crossover, since the quantised path is then a regression at the operating point.",
        options=("benchmarking at the production batch size", "reporting the crossover batch size with the claim"),
        tradeoff="whether the production batch size sits above the crossover point",
        flip="production concurrency falls below the crossover, at which point the quantised build is slower in the regime that matters",
        falsifier="the quantised build is faster at every batch size in the sweep including batch one",
        wrong_claim="The quantised build is slower in our benchmark, so quantisation does not help this model.",
        wrong_why="The benchmark ran at a batch size where the fixed dequantisation cost is charged against very few tokens, which is not the regime the service operates in.",
        threshold="Require any quantisation speedup claim to state the batch size it was measured at and the crossover point.",
        cost="A format rejected on a single-stream benchmark forgoes a real production saving; one accepted on the same benchmark ships a regression.",
        scaling="The crossover batch falls as weight count grows relative to per-token work, so larger models reach the benefit at smaller batches.",
        quant=q_dequant_overhead,
    ),
    Mechanism(
        key="weight_bytes_arithmetic", topic="quantization",
        title="the per-device weight footprint follows directly from parameter count, precision and shard degree",
        concepts=("quantization", "capacity_planning", "tensor_parallelism"),
        symptom="A precision change was proposed without anyone stating how much device memory it would actually free.",
        chain="Weight footprint per device is parameter count times bytes per parameter divided by the shard degree, so the memory freed by a precision change is exact arithmetic and does not need to be estimated by trial deployment.",
        metric="Per-device weight bytes computed from the configuration and reconciled against the runtime's reported allocation.",
        signature="Computed and reported footprints agree within the runtime's documented overhead, confirming the arithmetic covers the whole allocation.",
        confounders=(
            "Optimiser or adapter state held alongside the weights, which is not covered by the parameter count.",
            "Embedding and output layers kept at higher precision, which breaks the uniform bytes-per-parameter assumption.",
            "Allocator caching, which makes reported usage exceed live usage.",
        ),
        fixes=(
            "Compute the footprint from the configuration before deploying and reconcile it against the runtime's report.",
            "Account separately for layers deliberately kept at higher precision rather than applying one ratio.",
            "Re-derive the figure whenever the shard degree changes, since the two are coupled.",
        ),
        rollback="If computed and reported footprints disagree beyond documented overhead, stop using the computed value for admission sizing until the difference is explained.",
        options=("computing the footprint from configuration and reconciling it", "accounting separately for higher-precision layers"),
        tradeoff="whether the model applies one precision uniformly or keeps some layers wider",
        flip="a mixed-precision recipe is adopted, at which point a single bytes-per-parameter ratio stops describing the artifact",
        falsifier="computed and reported footprints differ by a constant ratio, which localises a missing term rather than an estimation error",
        wrong_claim="We will find out how much memory int8 frees once we deploy it and look at the allocator.",
        wrong_why="The quantity is exact arithmetic from parameter count, precision and shard degree, so deploying to discover it spends a deployment on a calculation.",
        threshold="Require the computed per-device footprint and the resulting cache budget before any precision change is scheduled.",
        cost="Deploying to measure a computable quantity spends accelerator time and a change window on arithmetic.",
        scaling="The absolute saving grows with parameter count while the shard degree divides it, so the two must be considered together rather than in sequence.",
        quant=q_weight_bytes_saved,
    ),
)
