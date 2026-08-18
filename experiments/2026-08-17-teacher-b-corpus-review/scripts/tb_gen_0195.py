import json

CORPUS="/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0195.jsonl"

SHARED = """
Shared mechanism, stated once so nothing below rests on hand-waving. Weight-only quantization (WOQ)
compresses the stored weight tensors and dequantizes them inside the GEMM epilogue. It reduces BYTES
OF WEIGHTS READ PER DECODE STEP and nothing else: it does not shrink the KV cache, does not reduce
attention FLOPs, and adds dequantization work on the compute path. The structural boundary follows
directly: WOQ can only pay in the memory-bandwidth-bound regime, i.e. small-batch long-decode
traffic. Prefill is compute bound, so TTFT is flat or mildly worse, and as batch size grows decode
GEMMs cross into the compute-bound regime and the speedup decays monotonically toward 1.0x.

Fair-comparison invariants, held identical across arms: same checkpoint lineage, tokenizer, prompt
set, sampling seeds, server build and flags except the quantization method, GPU SKU, driver and
clock policy, scheduler and max-batch parameters, KV dtype and max context. Only the weight numeric
format varies. Arms are interleaved in time (A/B/A/B) so thermal drift, background tenants and
driver state cannot be aliased into the treatment effect.

Rollback gate, pre-committed before any measurement: revert to the BF16 arm if (a) the paired
quality delta breaches the signed per-slice budget on any slice, (b) p95 end-to-end latency at
production concurrency regresses, (c) the measured GPU-seconds saving is below the pre-registered
threshold, or (d) any arm shows kernel fallback, numerical instability or NaN in a 24-hour soak.
Rollback is a config flip to an untouched BF16 deployment kept warm for the whole trial, not a
rebuild.
"""

S = [
 ("instruction-following-and-format-adherence-first",
  "aggregate accuracy is the wrong instrument; the first thing WOQ breaks is compliance with the prompt contract",
  """Falsifiable hypothesis H201: under greedy decoding on a fixed instruction-heavy prompt set, the
INT4 arm's format-adherence rate (exact schema match, required-section presence, length-bound
compliance, refusal-when-asked) stays within 0.5 absolute points of BF16. If adherence drops by
more than that while aggregate task scores are flat, the claim "quality preserved" is falsified and
must be restated as "preserved on scored tasks, degraded on instruction compliance".

Mechanism and why this is separable from accuracy. Instruction following is carried by
low-margin logit decisions at a small number of control tokens: the opening brace of a JSON object,
a stop token, a section header, a refusal prefix. Rounding error from group-wise weight
quantization perturbs logits by a bounded but non-uniform amount; where the BF16 margin between the
top two candidates is already small, a small perturbation flips the argmax. Scored benchmarks
average over many tokens and hide this, because a single flipped control token can void an entire
structured response while costing almost nothing in token-level accuracy.

Boundary condition that decides the experiment: the effect scales with how many control-token
decisions a response contains, not with response length in general. A 20-field JSON emission has
roughly 20 independent opportunities to break; free-form prose has almost none. So the comparison
must be run on the production prompt mix weighted by structured-output share, not on a benchmark
suite chosen for score comparability.

Instrumentation. Log, per arm and per prompt: schema-validation pass/fail with the raw failing
output retained, top-1 token agreement against BF16 under identical greedy decoding, and the
per-position logit margin distribution at control-token positions. Agreement is the sensitive
early-warning statistic; validity is the operational one.

Numbers, ESTIMATE with derivation. If per-call schema validity drops from 99.4% to 98.4%, a
10-step agent loop's end-to-end success falls from 0.994^10 = 94.2% to 0.984^10 = 85.1%
(ESTIMATE, derived by independence across steps, which is itself an assumption to be checked since
errors correlate within a session). A one-point per-call regression is therefore a nine-point
product regression, which is why aggregate parity cannot be the release gate.""",
  ["source_assistant restates a grading rubric instead of performing the comparison, so training on it teaches rubric recitation rather than analysis",
   "instruction-following and structured-output validity are absent from the listed metrics, so the most operationally damaging WOQ failure mode is never measured",
   "aggregate scores are treated as the decision statistic, allowing per-slice compliance regressions to pass unnoticed",
   "no top-1 agreement or logit-margin instrumentation, so early-warning signal is unavailable and only post-incident detection remains",
   "agent-loop compounding of per-call validity loss is not modelled, so the product-level blast radius of a small regression is unstated",
   "no retention of raw failing generations, making post-hoc triage of a compliance regression impossible"],
  ["per-arm schema-validation pass rate on the production-weighted prompt mix with all failing generations retained verbatim",
   "top-1 token agreement rate against the BF16 arm under identical greedy decoding and identical batch composition, reported per prompt class",
   "logit-margin distribution at control-token positions (JSON delimiters, stop tokens, section headers, refusal prefixes) for both arms",
   "measured share of production traffic that requires structured output, used to weight the prompt mix, sourced from serving telemetry not from assumption",
   "agent-loop end-to-end success rate measured directly on replayed multi-step sessions rather than inferred from a per-call independence model",
   "deterministic-decode configuration record (fixed batch composition, disabled continuous batching for the quality arm) proving agreement numbers are not batching artefacts)"],
  "rewrite", {"technical_correctness":3,"instruction_coverage":2,"operational_safety":3}, 0.63),

 ("tokenizer-and-detokenizer-invariance-first",
  "before comparing model quality, prove both arms consume and emit identical token streams",
  """Falsifiable hypothesis H202: for every prompt in the evaluation set, the BF16 and INT4 arms
produce byte-identical tokenizations of the input and use identical special-token and chat-template
handling, verified by hashing the token-id sequences. If the hashes differ for any prompt, all
downstream quality and latency deltas are void because the arms did not see the same input.

Why this is the first check and not a footnote. Quantized checkpoints are usually produced by a
conversion tool that rewrites the config and sometimes the tokenizer files, generation config, and
chat template. A changed EOS id, an added BOS, a different template whitespace convention, or a
different default max-new-tokens silently changes both the measured quality and the measured token
counts, and token counts are the denominator of the cost claim. This failure is common precisely
because it is invisible: both arms run, both produce plausible output, and the delta is attributed
to precision.

Verification procedure, executed before any benchmark run. Hash the tokenizer files and the chat
template of both artefacts and diff them field by field. For a fixed prompt set, dump input token
ids from each server and compare hashes. Compare generation configs including EOS/stop token sets,
repetition and length defaults, and any server-side prompt prefixing. Assert the detokenizer round
trip is identity on a sample of outputs from both arms.

Boundary condition. Output-token accounting must also be verified, because cost per 1,000 output
tokens changes if one arm counts special tokens and the other does not, and because a differing
stop-token set changes generation length. Length differences of a few percent are the same order as
the effect being measured.

Numbers, ESTIMATE with derivation. If the INT4 arm's stop handling yields 3% longer mean output,
GPU-seconds per request rise by roughly 3% at unchanged per-token cost (ESTIMATE, derived from
decode time being approximately linear in output length in the bandwidth-bound regime). Against a
target saving of 20-30% that is a tenth of the effect arriving purely from configuration drift.""",
  ["source_assistant lists grading criteria rather than executing the comparison, so it teaches rubric restatement",
   "tokenizer, chat template and generation-config drift introduced by the quantization conversion tool is not checked, so arms may not receive identical inputs",
   "output-token accounting differences change the cost denominator and are unverified",
   "no artefact-level diff of config files, so a silent template change is attributed to precision",
   "detokenizer round-trip identity is not asserted, allowing encoding differences to appear as quality differences",
   "stop-token set differences can change generation length by the same order as the effect under test"],
  ["sha256 of tokenizer files, chat template and generation config for both artefacts with a field-by-field diff",
   "hashes of input token-id sequences produced by each server for the identical prompt set, proving byte-identical tokenization",
   "per-arm stop-token and EOS set dump taken from the running server rather than from the repository files",
   "measured mean and tail output-token counts per arm on the same prompts, with special-token counting policy stated explicitly",
   "detokenizer round-trip identity check on a sampled set of generations from both arms",
   "conversion-tool name, version and full command line recorded in the run manifest"],
  "rewrite", {"technical_correctness":3,"instruction_coverage":2,"operational_safety":3}, 0.62),

 ("speculative-decoding-interaction-first",
  "WOQ and speculative decoding both target decode; measuring them apart predicts a benefit neither delivers together",
  """Falsifiable hypothesis H203: the GPU-seconds saving from INT4 measured with speculative decoding
DISABLED overstates the saving measured with the production speculative configuration ENABLED by
more than 30% relative. If the two savings are statistically indistinguishable, the interaction is
negligible and the simpler single-factor study is adequate; if they differ, any decision taken on
the spec-off number is invalid for a fleet that runs spec-on.

Mechanism of the interaction. Speculative decoding verifies K draft tokens in a single target
forward pass, so weights are read once per K accepted tokens instead of once per token. That
directly raises arithmetic intensity of the target step and shrinks the weight-bandwidth fraction f
that WOQ acts on. The two techniques are therefore substitutes for the same bottleneck, not
complements. A team that measures WOQ on a spec-off baseline and then deploys onto a spec-on fleet
will find most of the projected saving absent.

Second-order effects that must be controlled. Quantizing the target model changes its logit
distribution, which changes draft acceptance rate; a lower acceptance rate reduces speculative
efficiency and can make the quantized arm worse end to end even where its per-step cost is lower.
Acceptance rate must therefore be measured per arm, not assumed constant. If the draft model is
also quantized, that is a second variable and needs its own arm.

Experimental design. Four arms at minimum: BF16 spec-off, INT4 spec-off, BF16 spec-on, INT4 spec-on,
all at the same SLO-feasible concurrency. Report acceptance rate and mean accepted length per
spec arm alongside the cost denominator.

Numbers, ESTIMATE with derivation. If mean accepted length is 2.5 tokens per target pass, weight
reads per output token fall by 2.5x before any quantization, so the residual bandwidth-bound
fraction available to WOQ is roughly 1/2.5 of the spec-off case (ESTIMATE, derived from weight
bytes per output token scaling inversely with accepted length). An acceptance-rate drop that moves
mean accepted length from 2.5 to 2.2 costs about 12% of decode throughput (ESTIMATE, derived from
the ratio 2.2/2.5), which can exceed the WOQ gain remaining at that operating point.""",
  ["source_assistant supplies a rubric rather than an analysis, teaching restatement instead of reasoning",
   "interaction with speculative decoding is omitted although both techniques contend for the same decode bottleneck",
   "draft acceptance rate is not measured per arm, so a quantization-induced acceptance regression is invisible",
   "results measured on a spec-off baseline may be transplanted onto a spec-on production fleet, overstating savings",
   "if the draft model is also quantized the study silently varies two factors at once",
   "no SLO-matched operating point per arm, so throughput comparisons are taken at incomparable latency"],
  ["four-arm measurement (BF16/INT4 x spec-off/spec-on) at SLO-feasible concurrency with GPU-seconds per 1,000 output tokens reported per arm",
   "measured draft acceptance rate and mean accepted length per speculative arm, with confidence intervals over the request trace",
   "draft-model identity, precision and hash recorded per arm, proving the draft was held constant across target-precision arms",
   "production configuration record showing whether the deployed fleet runs speculative decoding and with what parameters",
   "per-phase timing breakdown separating draft generation, target verification and rejection-rollback cost",
   "profiler trace confirming the quantized fused kernel is dispatched in the target verification pass, not only in single-token decode"],
  "rewrite", {"technical_correctness":3,"instruction_coverage":2,"operational_safety":3}, 0.61),

 ("quantization-scheme-taxonomy-first",
  "'INT4 weight-only' names a family, not a configuration; the comparison must state which member is under test",
  """Falsifiable hypothesis H204: within the same nominal bit width, the spread in eval-suite score
across scheme variants (group size 32 vs 128 vs per-channel; symmetric vs asymmetric; act-order on
vs off; outlier channels retained in higher precision vs not) exceeds the BF16-to-INT4 gap of the
default variant. If it does, then reporting a single "INT4" result is not a measurement of
quantization at all, and the scheme parameters must be treated as first-class independent variables.

Mechanism. Group-wise quantization fits one scale (and optionally one zero point) per group of
weights along the reduction dimension. Smaller groups track local dynamic range better and cost more
metadata; asymmetric schemes handle skewed distributions but need a zero point; activation-order
permutations change which weights share a group, which matters exactly when outlier channels exist.
Each knob trades accuracy against memory and against kernel availability, and the kernels available
differ per variant, so accuracy and speed are not independently choosable.

Boundary conditions that must be checked, not assumed. Group size must divide the per-shard input
dimension; a tensor-parallel degree change can therefore invalidate a scheme. Some variants have no
fused kernel on some architectures and silently fall back. Metadata overhead grows as group size
shrinks and eats the memory saving that motivated the project.

Design. A scheme sweep run before the headline comparison, reporting for each variant: eval score,
weight bytes including metadata, dispatched kernel names, and decode throughput at the target
concurrency. The headline arm is then the Pareto-selected variant, named explicitly in the report.

Numbers, ESTIMATE with derivation. For 4-bit weights with FP16 scale and 4-bit zero per group,
metadata per weight is (16+4)/group_size bits. At group 128 that is 0.156 bits/weight, giving 4.16
bits/weight total, about 4% over nominal; at group 32 it is 0.625 bits/weight, giving 4.63
bits/weight, about 16% over nominal (both ESTIMATE, derived directly from the formula). So the
memory saving claimed from "4-bit" is materially wrong at small group sizes and must be recomputed
per variant from measured artefact size.""",
  ["source_assistant is a grading rubric, not an executed comparison",
   "treats 'weight-only quantization' as a single configuration when scheme parameters can dominate the precision effect",
   "metadata overhead from scales and zero points is ignored, so claimed memory savings are overstated at small group sizes",
   "kernel availability differs per scheme variant, coupling accuracy choices to performance in ways the design does not capture",
   "group-size divisibility constraints interact with tensor-parallel degree and can silently disable fused kernels",
   "no Pareto selection step, so the reported variant may be an arbitrary tool default presented as 'INT4'"],
  ["scheme sweep table reporting eval score, measured artefact bytes including metadata, dispatched kernel names and decode throughput for each variant",
   "measured on-disk and in-GPU weight bytes per variant compared against the analytic bits-per-weight formula",
   "profiler kernel-name histogram per variant proving fused-kernel dispatch rather than dequant-then-GEMM fallback",
   "record of tensor-parallel degree and per-shard input dimensions demonstrating group-size divisibility for every quantized layer",
   "explicit list of layers excluded from quantization (lm_head, first/last block) per variant",
   "quantization tool version, full parameter set and artefact hash for every variant in the sweep"],
  "rewrite", {"technical_correctness":3,"instruction_coverage":2,"operational_safety":3}, 0.62),

 ("engine-version-and-dependency-drift-first",
  "the comparison has a shelf life; an engine bump can invert the result without any model change",
  """Falsifiable hypothesis H205: re-running both arms unchanged after the next scheduled engine,
kernel-library or driver upgrade reproduces the original GPU-seconds delta within the pre-declared
run-to-run band. If the delta moves outside that band on a pure dependency bump, the result is a
property of one software stack and must be re-certified per upgrade rather than treated as a
standing conclusion.

Mechanism. The WOQ benefit is delivered by kernels, and kernel selection is a runtime decision made
by heuristics that change between releases. A patch-level upgrade can add a faster BF16 path (which
shrinks the quantized arm's relative win), add or remove a fused low-precision kernel for a
particular shape, change autotuning defaults, or change scheduler behaviour that shifts the
prefill/decode mix. None of these are announced as performance changes to your specific shapes.

Operational consequence that the design must absorb. If quantization is adopted, the fleet now runs
two numeric paths, and every dependency upgrade must be validated on both. That is a recurring
engineering cost and a recurring risk of asymmetric regression, where the upgrade is safe on the
baseline and harmful on the quantized arm.

Control procedure. Pin the entire stack by digest for the study, record engine commit, kernel
library, driver and firmware versions, and container image digest per arm. Establish a re-run job
that repeats a short canonical benchmark on every dependency change and alerts on delta movement
beyond the band. Keep the BF16 artefact and its serving configuration continuously deployable.

Numbers, ESTIMATE with derivation. If dependency bumps occur monthly and re-certification costs 8
engineer-hours of benchmark plus quality re-check per bump, that is roughly 96 engineer-hours per
year (ESTIMATE, derived from cadence times per-bump effort) charged against the saving. At an
internal rate of 150 USD/hour that is about 14,400 USD/year (ESTIMATE), which must appear in the
cost model beside the GPU savings or the business case is incomplete.""",
  ["source_assistant provides grading criteria rather than an analysis of the problem",
   "no version pinning or dependency-drift control, so the result is undated and unattributable to a stack",
   "the recurring re-certification cost of maintaining two numeric paths is absent from the cost framing",
   "kernel-selection heuristics can change on patch upgrades and invert the comparison without any model change",
   "asymmetric regression risk (upgrade safe on baseline, harmful on quantized arm) is not covered by any monitoring",
   "no continuous re-run job, so drift is discovered by incident rather than by test"],
  ["per-arm record of engine commit SHA, kernel-library version, CUDA/driver and firmware versions, and container image digest",
   "repeat measurement of both arms after a dependency upgrade with the delta compared against the pre-declared run-to-run band",
   "profiler kernel-name histogram captured before and after the upgrade showing whether dispatch changed",
   "scheduled canonical benchmark job configuration committed to version control, with its alerting thresholds",
   "engineering-time log for each re-certification cycle, feeding the amortized cost term in the cost model",
   "evidence that the BF16 artefact and configuration remain continuously deployable, including a dated redeploy drill"],
  "rewrite", {"technical_correctness":3,"instruction_coverage":2,"operational_safety":3}, 0.61),

 ("energy-and-power-envelope-first",
  "if the claim is cost, measure joules per token, because power capping and clocks silently set the answer",
  """Falsifiable hypothesis H206: the INT4 arm reduces energy per 1,000 output tokens at the fixed
SLO by at least as much as it reduces GPU-seconds. If energy savings are materially smaller than
time savings, the arm is running at higher average power and the cost model based on time alone
overstates the benefit wherever power or cooling is the binding constraint.

Mechanism. Bandwidth-bound decode leaves compute units idle and draws less power than a
compute-bound phase. WOQ shifts the step toward compute (dequantization plus GEMM at higher
arithmetic intensity per byte read), so it can raise average power while lowering wall time. In a
power-capped or thermally-limited rack, wall-clock savings that raise power do not convert into
capacity, because the cap forces clock reduction and the measured speedup shrinks or disappears at
scale even though it was real on an unconstrained single node.

Confound to control explicitly. Clock policy must be identical and preferably locked across arms.
With auto-boost, a short benchmark can run at a boost clock that a sustained production load cannot
hold, so the quantized arm's advantage measured in a two-minute run may not survive a one-hour soak.
Measurement therefore requires a soak long enough to reach thermal steady state, with the warmup
window discarded and the steady-state window verified flat.

Instrumentation. Sample board power, SM and memory clocks, temperature and any clock-throttle reason
codes at fixed intervals for both arms, and integrate power over the measurement window to obtain
joules per 1,000 output tokens. Report both the unconstrained and the power-capped operating points
if the fleet runs capped.

Numbers, ESTIMATE with derivation. If BF16 draws 220 W average during decode and INT4 draws 250 W
while cutting wall time 25%, energy per token changes by 0.75 x (250/220) = 0.85, a 15% energy
saving against a 25% time saving (ESTIMATE, derived from the product of time ratio and power ratio).
Under a hard power cap the realizable capacity gain tracks the 15%, not the 25%.""",
  ["source_assistant is a rubric restatement rather than an analysis of the cost claim",
   "energy and power are absent although the stated goal is cost and power is often the binding constraint",
   "clock policy and boost behaviour are uncontrolled, so short-run measurements may not survive sustained load",
   "no thermal steady-state requirement, so warmup transients can be reported as steady-state performance",
   "power-capped fleet behaviour is not distinguished from unconstrained single-node behaviour",
   "throttle reason codes are not collected, so a capped run can be misread as a kernel or precision effect"],
  ["board power, SM and memory clock, temperature and throttle-reason-code time series sampled at fixed intervals for both arms over a soak reaching thermal steady state",
   "integrated joules per 1,000 output tokens per arm at the SLO-feasible operating point, with the discarded warmup window declared",
   "clock policy record (locked clocks or documented auto-boost) identical across arms, captured from the device rather than from intent",
   "measurements repeated at the production power cap where the fleet runs capped, reported separately from the uncapped point",
   "steady-state window flatness check on the metric time series, with the tolerance stated in advance",
   "rack-level or facility-level power and cooling constraint documentation showing whether power or time is the binding capacity term"],
  "rewrite", {"technical_correctness":3,"instruction_coverage":2,"operational_safety":3}, 0.60),

 ("multi-tenant-interference-and-isolation-first",
  "a fair comparison on an isolated node can be an unfair prediction for a shared cluster",
  """Falsifiable hypothesis H207: the INT4 arm's advantage measured on a dedicated node persists,
within the pre-declared band, when both arms are re-measured under the production co-tenancy pattern
(neighbouring jobs on the same host, shared PCIe/NVLink paths, shared storage and network). If the
advantage shrinks materially under co-tenancy, the isolated-node result is not a valid basis for a
fleet decision.

Mechanism. WOQ's benefit is a reduction in memory-bandwidth demand. In a shared environment, the
resources that contend are exactly the ones the benefit is denominated in: HBM bandwidth is private
per GPU, but host memory bandwidth, PCIe, NVLink and network are shared, and a noisy neighbour that
saturates them changes the phase mix. Weight loading, KV offload paths, and any host-side
detokenization or logging contend for host resources whose availability differs between an isolated
benchmark and a packed host.

Design. Measure both arms twice: dedicated node, and under a synthetic or replayed co-tenancy load
that reproduces production host packing. Randomize arm order and interleave, because interference is
time-varying and back-to-back runs alias it into the treatment effect. Report per-arm variance under
co-tenancy separately from the mean, since interference typically widens the tail more than it moves
the median, and the SLO is defined on the tail.

Isolation controls to record. CPU pinning and NUMA placement, GPU-to-NUMA affinity, cgroup limits,
number of co-located containers, and whether MIG or time-slicing is in use. MPS or time-slicing in
particular changes the effective bandwidth available per arm and must be identical across arms.

Numbers, ESTIMATE with derivation. If co-tenancy inflates p99 TPOT by 20% on both arms but the
quantized arm's saving is 25%, the SLO-feasible concurrency can still shift adversely because the
tail, not the mean, sets admission (ESTIMATE; the magnitude must be MEASURED because interference
distributions are workload-specific and not analytically predictable).""",
  ["source_assistant restates grading criteria instead of analysing the comparison",
   "measurements on an isolated node are implicitly assumed to transfer to a shared production cluster",
   "host-side contention (PCIe, NUMA, host memory bandwidth, storage, network) is not controlled or recorded",
   "tail-latency inflation under co-tenancy is not separated from mean shifts although the SLO is defined on the tail",
   "MIG, MPS or time-slicing configuration can differ between arms and change effective bandwidth",
   "run order is not randomized or interleaved, so time-varying interference is aliased into the treatment effect"],
  ["paired measurements of both arms on a dedicated node and under replayed production co-tenancy, with interleaved randomized run order",
   "per-arm latency distributions under co-tenancy reporting p50/p95/p99 and variance, not means alone",
   "host placement record: CPU pinning, NUMA affinity, cgroup limits, co-located container count and GPU partitioning mode (MIG/MPS/time-slicing) identical across arms",
   "host-side resource telemetry (PCIe throughput, host memory bandwidth, storage and network utilisation) sampled during both arms",
   "SLO-feasible concurrency recomputed under co-tenancy for each arm rather than carried over from the isolated measurement",
   "production host-packing statistics used to construct the co-tenancy load, sourced from cluster telemetry"],
  "rewrite", {"technical_correctness":3,"instruction_coverage":2,"operational_safety":3}, 0.61),

 ("artefact-conversion-integrity-first",
  "the quantized checkpoint is a new binary artefact and must clear an integrity gate before it is benchmarked",
  """Falsifiable hypothesis H208: the quantized artefact, loaded by the production serving path,
reproduces the quantization tool's own reported per-layer reconstruction error within tolerance, and
its layer inventory matches the base model exactly in shape, count and naming. If any layer is
missing, mis-shaped, silently left in high precision, or loaded with a different config than the one
recorded, the artefact is not the thing that was validated and every downstream number is void.

Mechanism of the failure. Quantization is a conversion pipeline: load, calibrate, pack, serialize,
then reload under a different runtime. Each stage can drop or alter state. Common concrete failures
are layers skipped because a shape did not match the group size, a config field the serving engine
ignores so it loads with different defaults than the packer wrote, and a partially-written artefact
that loads without error because the format tolerates missing optional tensors. None of these raise
an exception; they produce a model that runs and is quietly wrong or quietly unquantized.

Integrity gate, executed before benchmarking. Verify artefact hash against the publication record.
Enumerate every linear layer in the loaded server and assert its stored dtype, group size and
packing match the intended scheme; list deliberately excluded layers explicitly. Compare
per-layer output against the BF16 reference on a fixed activation batch and check the error against
the packer's reported reconstruction error. Confirm the loaded quantization config as reported by
the running server, not as written in the file.

Supply-chain requirement. Record base checkpoint identity, licence and hash; quantization tool and
version; calibration corpus hash and rights status; build environment digest; and enforce
signature or hash verification at load time through an audited publication path.

Numbers, ESTIMATE with derivation. If 4 of 40 blocks silently remain in BF16, weight bytes land
about 10% above the projected figure and the decode speedup ceiling falls from 3.8x to roughly 2.9x
(ESTIMATE, derived from weighted average bytes per parameter across quantized and unquantized
blocks). That is a large fraction of the effect being measured, arriving from a packaging defect.""",
  ["source_assistant is a rubric, not an analysis, and teaches restatement of grading criteria",
   "artefact integrity is unverified, so a partially quantized or mis-loaded checkpoint can be benchmarked as if intended",
   "loaded runtime configuration is not read back from the running server, allowing silent default substitution",
   "no per-layer reconstruction-error cross-check against the quantization tool's own report",
   "provenance, licence and load-time integrity verification of the artefact are entirely absent",
   "silently unquantized layers shift both memory and speedup ceilings by amounts comparable to the effect under test"],
  ["sha256 of base and quantized artefacts matched against the publication record, with signature or hash verification enforced at load time",
   "per-layer inventory dumped from the running server listing stored dtype, group size and packing for every linear layer, plus the explicit exclusion list",
   "measured per-layer output error against the BF16 reference on a fixed activation batch, compared with the quantization tool's reported reconstruction error",
   "quantization config as reported by the running server, diffed against the config written by the packer",
   "measured in-GPU weight bytes compared against the analytic projection for the intended scheme",
   "provenance chain: base checkpoint licence and hash, tool version and parameters, calibration corpus hash and rights status, build environment digest, audited publication path"],
  "rewrite", {"technical_correctness":3,"instruction_coverage":2,"operational_safety":3}, 0.62),

 ("admission-control-and-queueing-first",
  "the SLO is set by the queue, not the kernel; compare arms at their admission limits or not at all",
  """Falsifiable hypothesis H209: at the concurrency where each arm just satisfies the p95 SLO, the
INT4 arm admits at least 20% more requests per second than BF16. If instead the gain appears only at
concurrencies where p95 is already out of contract, the throughput win is unclaimable and the correct
conclusion is that the arm improves saturated throughput but not SLO-bounded capacity.

Mechanism. End-to-end latency is queueing delay plus service time. WOQ reduces per-step service time
in the bandwidth-bound regime, but as offered load rises the system approaches its stability
boundary and queueing delay dominates; near saturation latency is governed by utilisation, not by
per-step cost. So a modest service-time improvement can produce a large capacity gain (by moving the
stability boundary) or almost none (if the admission controller, KV block limit, or max-num-seqs cap
binds first). Which of these holds is a property of the scheduler configuration, not of the kernel.

Design. For each arm, sweep offered load with an open-loop generator using a production-derived
arrival process, find the highest sustainable rate meeting the p95 SLO, and read GPU-seconds per
1,000 output tokens at that point. Closed-loop generators are also run, but only same-generator
comparisons are reported, because closed loop suppresses queueing by construction and systematically
flatters the slower arm.

Binding-constraint identification, which is the actual deliverable. Record, per arm at the SLO
point, which limit is active: KV blocks exhausted, max-num-seqs reached, scheduler preemption rate
nonzero, or pure compute saturation. If the KV block limit binds, the gain is a capacity effect from
freed memory and must be reported as such; if preemption is occurring, the comparison is measuring
scheduler policy rather than precision.

Numbers, ESTIMATE with derivation. Under an M/M/1-style approximation, reducing mean service time by
25% raises the sustainable arrival rate at fixed mean latency by roughly 33% (ESTIMATE, derived from
rate scaling inversely with service time at fixed utilisation); real LLM serving with batching and
heavy-tailed output lengths will fall short of this, so the analytic figure is an upper bound to be
falsified by measurement, not a prediction.""",
  ["source_assistant lists grading criteria and does not perform the comparison",
   "throughput is treated as comparable without fixing an SLO, so gains may be reported at latencies the contract forbids",
   "queueing dynamics and the admission-control binding constraint are absent, so kernel effects and capacity effects are conflated",
   "closed-loop versus open-loop load generation is unspecified, and closed loop systematically flatters the slower arm",
   "scheduler preemption and KV block exhaustion are not recorded, so scheduler policy can be reported as a precision effect",
   "arrival process is not derived from production telemetry, so burstiness and tail behaviour may be unrepresentative"],
  ["per-arm load sweep with an open-loop production-derived arrival process, reporting the highest sustainable rate meeting the p95 SLO and GPU-seconds per 1,000 output tokens at that point",
   "binding-constraint record at the SLO point per arm: KV block occupancy, max-num-seqs, preemption rate and compute utilisation",
   "closed-loop and open-loop results reported separately with same-generator comparisons only",
   "production arrival-process characterisation (rate, burstiness, input/output length distributions including tails) sourced from serving telemetry",
   "clamped-KV arm isolating the capacity effect from the bandwidth effect, with admitted KV blocks matched to the BF16 arm",
   "per-request latency traces retained so percentiles and queueing delay can be recomputed by a reviewer"],
  "rewrite", {"technical_correctness":3,"instruction_coverage":2,"operational_safety":3}, 0.62),

 ("regression-triage-and-attribution-first",
  "design the study so that a failure is diagnosable, not just detectable",
  """Falsifiable hypothesis H210: given only the artefacts the study is designed to produce, an
engineer who did not run it can attribute any observed quality or latency regression to one of a
pre-enumerated set of causes (kernel fallback, calibration mismatch, tokenizer or config drift,
scheduler or capacity difference, workload shift, or genuine precision loss) without re-running the
experiment. If attribution requires a re-run, the study's instrumentation is inadequate and should be
fixed before the comparison is trusted, regardless of what the headline numbers say.

Why this is a design constraint and not a postscript. Quantization studies fail in ways that all
present as "the quantized arm is worse", and the remedies are mutually exclusive: a fallback needs a
kernel fix, a calibration mismatch needs more or different data, a config drift needs a packaging
fix, and genuine precision loss needs abandoning the bit width. Without discriminating evidence
collected during the run, the team iterates by guessing, which is expensive and tends to terminate
on whichever hypothesis was checked first rather than the true one.

Discriminating instrumentation, collected on every run. Kernel-name histogram with time attribution
(separates fallback). Per-slice paired quality deltas with calibration-represented and
calibration-absent slices labelled (separates calibration mismatch). Config and tokenizer hashes
plus loaded-config readback (separates drift). KV block occupancy, preemption counts and admitted
concurrency (separates capacity and scheduler effects). Request-mix statistics per measurement window
(separates workload shift). Per-layer output error against the BF16 reference (isolates genuine
numeric loss to specific layers).

Deviation and exclusion log. Every retune, restart or discarded run is recorded with a reason.
Silent exclusion of "bad runs" is the most common route by which a false positive survives review,
and it also destroys the evidence needed for triage.

Numbers, ESTIMATE with derivation. Adding the above instrumentation costs on the order of a few
percent of run wall time (profiler traces are sampled on short windows, not run continuously) and a
few gigabytes of trace storage for a multi-week trial at roughly 200 bytes per request record
(ESTIMATE, derived from request volume times record size). That is negligible against the cost of a
single misattributed iteration cycle.""",
  ["source_assistant restates a grading rubric instead of producing an analysis",
   "the study is designed to detect a difference but not to attribute it, so a regression cannot be diagnosed without re-running",
   "discriminating instrumentation for the main competing failure causes is absent",
   "no deviation or exclusion log, so discarded runs can silently bias the result and destroy triage evidence",
   "per-layer error localisation is missing, so genuine precision loss cannot be separated from packaging or kernel defects",
   "raw per-request traces are not retained, preventing independent re-slicing and percentile recomputation"],
  ["kernel-name histogram with time attribution captured on sampled windows of every run, per arm",
   "per-slice paired quality deltas with each slice labelled as calibration-represented or calibration-absent",
   "config, tokenizer and chat-template hashes plus loaded-configuration readback from the running server for both arms",
   "KV block occupancy, preemption counts and admitted concurrency time series per arm per measurement window",
   "request-mix statistics (input/output length distributions, request-class shares) per measurement window, used to detect workload shift",
   "per-layer output error against the BF16 reference on a fixed activation batch, localising numeric loss to specific layers",
   "deviation and exclusion log recording every retune, restart and discarded run with its reason, plus retained raw per-request latency traces"],
  "rewrite", {"technical_correctness":3,"instruction_coverage":2,"operational_safety":3}, 0.63),
]

corpus=[json.loads(l) for l in open(CORPUS) if l.strip()]
sl=corpus[1940:1950]
assert len(sl)==10 and len(S)==10

out=[]
for rec,(stance,tag,body,risks,ev,dec,qd,conf) in zip(sl,S):
    m={x['role']:x['content'] for x in rec['messages']}
    ca=("Analytical stance under test: %s - %s.\n\n%s\n%s" % (stance, tag, body.strip(), SHARED.strip()))
    out.append({
        "source_id": rec["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": dec,
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": ca,
        "quality_dimensions": qd,
        "risks": risks,
        "evidence_required": ev,
        "confidence": conf,
    })

with open(OUT,"w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False)+"\n")
print("WROTE", len(out), OUT)
print("ids", out[0]["source_id"], out[-1]["source_id"])
