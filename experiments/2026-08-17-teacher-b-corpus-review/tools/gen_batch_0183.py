import json
ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
START, N = 1820, 10
OUT = f"{EXP}/results/train-batch-0183.jsonl"

rows = [json.loads(l) for l in open(CORPUS) if l.strip()][START:START + N]

STANCE = "Analytical stance under test: "

BODIES = [
("W-only quantization is a memory-bandwidth trade, not a FLOPs trade; a fair comparison must therefore be run at fixed batch/concurrency points on the same kernel stack.",
 """Mechanism. Weight-only quantization (INT8/INT4 group-wise, AWQ/GPTQ-style) shrinks the weight tensor read per token. Decode is weight-bandwidth-bound at low-to-moderate batch, so the theoretical decode speedup ceiling is the weight-byte ratio (BF16->INT4 = 4x, ESTIMATE derived from 2 bytes/param -> 0.5 bytes/param plus group scales/zeros, which add roughly 128-bit metadata per group of 128 weights, i.e. ~3% overhead). Prefill is compute-bound and typically gets slower because weights must be dequantized to the compute dtype before the GEMM.
Fair-comparison protocol. (1) Freeze model checkpoint, tokenizer, sampling params (greedy for quality, fixed seed for sampled), max_model_len, and the serving engine version. (2) Vary exactly one factor: weight dtype. (3) Report TTFT, TPOT, end-to-end p50/p95/p99, output tokens/s, and goodput at matched SLO, swept over concurrency 1,2,4,8,16,32,64 rather than at one point.
Falsifiable hypothesis. H1: at concurrency<=8, INT4 weight-only reduces TPOT by >=35% relative to BF16 on the same GPU, with MMLU/GSM8K delta <=1.0 absolute point. H0: the reduction is <35% or quality delta >1.0 point.
Confounders to control. KV-cache dtype (must stay identical or the experiment measures two variables), CUDA graph capture on/off, chunked prefill settings, kernel availability (a missing fused INT4 kernel silently falls back to a dequant+BF16 GEMM path and erases the win), GPU clock throttling, and the calibration set leaking into the eval set.
Evidence. Kernel-level trace (Nsight Systems) proving the quantized GEMM path is actually taken; nvidia-smi power/clock logs across both runs; per-request latency CDFs, not means; calibration-set provenance hash.
Rollback gate. Revert to BF16 if quality delta exceeds 1.0 absolute point on the agreed suite, if p99 TTFT regresses >20%, or if any kernel fallback is observed in the trace.""",
 "rewrite", {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.74,
 ["Source answer lists metric names without mechanism, so it cannot distinguish a real kernel win from a dequant fallback",
  "No concurrency sweep means a single-point measurement can be cherry-picked",
  "Prefill regression risk is unmentioned"],
 ["Nsight kernel trace confirming quantized GEMM dispatch", "Latency CDF per concurrency level", "Calibration-set/eval-set disjointness proof"]),

("The comparison is only fair if quality is measured with a pre-registered suite and a pre-registered decision rule, otherwise the metric is chosen after seeing the result.",
 """Mechanism. Weight-only quantization perturbs each weight by a bounded rounding error; the induced logit perturbation compounds through depth, so degradation is task-dependent and concentrates on long-chain reasoning and rare-token tasks rather than on short classification.
Pre-registration protocol. Before running the quantized model, write down: the eval suite (e.g. MMLU, GSM8K, HumanEval, plus a domain-held-out set), the number of samples per task, the sampling config, the acceptance threshold, and the statistical test. Any metric added after seeing results must be reported separately as exploratory.
Falsifiable hypothesis. H1: the paired per-item accuracy difference between BF16 and INT4 has a 95% bootstrap CI whose upper bound of degradation is <=1.5 absolute points. H0: the CI includes degradation >1.5 points.
Statistics. Use paired per-item comparison on identical prompts, not two independent aggregate scores; with n=1000 items and per-item accuracy near 0.7, the ESTIMATE of the aggregate standard error is ~1.4 points (derived from sqrt(0.7*0.3/1000)), so an unpaired design cannot resolve a 1-point effect. Pairing removes prompt-difficulty variance and is what makes the test powered.
Confounders. Non-determinism from batching (continuous batching changes reduction order and thus logits); pin batch composition or run with a fixed batch size and deterministic kernels for the quality arm.
Evidence. Pre-registration document with a timestamp/commit hash; raw per-item generations for both arms; bootstrap CI script and its seed.
Rollback gate. Revert if the paired CI upper bound exceeds 1.5 points, or if any single safety/refusal-behaviour probe flips.""",
 "rewrite", {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.72,
 ["'Report confidence intervals' without specifying paired design yields an underpowered test",
  "Post-hoc metric selection can manufacture a favourable result",
  "Continuous batching non-determinism can be mistaken for quantization damage"],
 ["Timestamped pre-registration commit", "Per-item paired generations for both arms", "Bootstrap CI with recorded seed"]),

("Cost is the actual claim being made, so the comparison must be denominated in currency per served token at a fixed SLO, not in raw throughput.",
 """Mechanism. Weight-only quantization lowers cost through two distinct channels that must not be conflated: (a) fewer GPUs needed to hold the weights, and (b) higher tokens/s per GPU. Channel (a) is a step function (you either fit on 1 GPU or you do not); channel (b) is continuous and only materialises at the concurrency your traffic actually has.
Cost model. cost_per_1M_tokens = (GPU_hourly_rate * replicas) / (goodput_tokens_per_s * 3600) * 1e6. Every term must be measured under the same SLO definition (e.g. p95 TTFT <= 500 ms, p95 TPOT <= 40 ms). Reporting saturated throughput without an SLO gate is the single most common way this comparison is rigged.
Falsifiable hypothesis. H1: at the production traffic mix replayed from real traces, INT4 reduces cost per 1M output tokens by >=25% at equal SLO attainment. H0: the reduction is <25%.
Numbers. If INT4 removes one of two GPUs, the ESTIMATE of savings is ~50% from channel (a) alone, derived purely from replica count and independent of throughput; that is why channel decomposition matters for attribution.
Confounders. Traffic shape (input/output length distribution) dominates; a synthetic uniform-length benchmark will over-report the quantization win because decode-heavy synthetic loads maximise the bandwidth benefit. Also control for spot vs on-demand pricing and for tensor-parallel degree changes.
Evidence. Production trace replay with the real length distribution; SLO attainment percentage per arm; per-arm replica count and instance type; billing export cross-check.
Rollback gate. Revert if SLO attainment drops below the current baseline percentage, or if measured savings are under 25% once channel (a) is excluded.""",
 "rewrite", {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.73,
 ["Throughput without an SLO gate overstates savings",
  "Synthetic uniform-length load biases the result toward quantization",
  "Replica-count savings can be double-counted with throughput savings"],
 ["Production trace replay artefacts", "Per-arm SLO attainment log", "Cloud billing export for both arms"]),

("Most reported quantization wins are kernel-coverage artefacts: the correct first experiment is a dispatch audit, before any latency number is collected.",
 """Mechanism. A weight-only quantized checkpoint only accelerates inference if the runtime has a fused dequant-GEMM kernel for the exact (dtype, group size, shape, GPU arch) tuple. When coverage is missing, engines fall back to materialising BF16 weights on the fly, which is strictly slower than BF16 and still costs the quantization quality loss.
Dispatch audit procedure. Run one short generation with kernel-level profiling and enumerate the GEMM kernels executed per layer. Confirm the quantized kernel name appears for every projection (qkv, o, gate/up, down). Any layer running a dequant+standard GEMM path is an unquantized layer for performance purposes even though it is quantized on disk. Also record which layers were deliberately kept in high precision (commonly lm_head and sometimes the first/last block).
Falsifiable hypothesis. H1: >=95% of linear-layer GEMM time is spent in fused quantized kernels. H0: it is lower, in which case any latency comparison measures the fallback path, not quantization. If only 80% of GEMM time is fused, the ESTIMATE of the achievable decode speedup collapses from ~3x to ~1.6x, derived from Amdahl's law applied to the unfused 20% time fraction that instead pays an extra dequantization pass.
Boundary conditions. Group size must divide the input dimension; odd tensor-parallel shardings can break this and silently disable fused kernels. Arch matters: kernels tuned for one generation may not exist on another.
Evidence. Kernel-name histogram with time attribution; the engine's quantization config as loaded at runtime (not as written in the file); a list of layers excluded from quantization.
Rollback gate. Do not proceed to the cost/quality comparison at all until the dispatch audit passes; if coverage cannot reach the threshold, the honest conclusion is 'not supported on this stack', not 'quantization did not help'.""",
 "rewrite", {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.76,
 ["Silent kernel fallback makes a quantized run slower while still losing quality",
  "Tensor-parallel sharding can break group-size divisibility",
  "Runtime config may differ from the on-disk config"],
 ["Kernel-name histogram with time attribution", "Runtime-loaded quantization config dump", "Per-layer quantization exclusion list"]),

("Memory savings on weights are frequently cancelled by KV cache, so the fair comparison must report the full memory budget decomposition, not just checkpoint size.",
 """Mechanism. Total GPU memory = weights + KV cache + activations + fragmentation/reserve. Weight-only quantization shrinks only the first term. The freed memory usually gets reallocated to KV cache, which raises the maximum concurrency; that is the real serving benefit, and it is invisible if you only report 'model size dropped'.
Decomposition protocol. For each arm, report weight bytes, KV bytes per token per sequence, the engine's allocated KV block count, peak activation memory at max batch, and the reserved fraction. KV bytes per token = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes. For a 70B-class model with 80 layers, 8 KV heads, head_dim 128 in FP16 the ESTIMATE is 2*80*8*128*2 = 327,680 bytes/token = ~0.31 MiB/token, derived directly from that formula; a 8k-token sequence therefore costs ~2.5 GiB of KV.
Falsifiable hypothesis. H1: INT4 weights increase the maximum concurrent sequences at fixed context length by >=2x versus BF16 on the same GPU count. H0: the increase is <2x.
Confounders. KV dtype and prefix caching must be held constant; enabling FP8 KV in only one arm changes two variables. Fragmentation differs between arms and must be read from the allocator, not inferred.
Evidence. Allocator snapshot per arm; engine startup log reporting KV block count; peak-memory profile at max batch; OOM-boundary bisection.
Rollback gate. Revert if the OOM boundary is not at least the previously certified concurrency, or if fragmentation forces reserve above the operational safety margin.""",
 "rewrite", {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.72,
 ["Reporting only checkpoint size hides that KV cache dominates memory at long context",
  "Changing KV dtype together with weight dtype confounds the experiment",
  "Fragmentation can consume the nominal savings"],
 ["Per-arm allocator snapshot", "Engine KV block-count startup log", "OOM-boundary bisection record"]),

("Calibration data is a hidden independent variable; without freezing and disclosing it the comparison is not reproducible and is trivially gameable.",
 """Mechanism. GPTQ/AWQ-style methods choose scales and clipping ranges using a calibration corpus. The resulting quality is a function of that corpus's domain match, sequence length, and sample count. Two 'INT4' checkpoints of the same model can differ by more than the BF16-to-INT4 gap itself purely from calibration choices.
Protocol. Fix calibration corpus identity (content hash), sample count, sequence length, and random seed. Run at least three calibration seeds and report the spread, not a single run. Verify the calibration corpus is disjoint from every eval set by exact and near-duplicate matching.
Falsifiable hypothesis. H1: across three calibration seeds, the eval-suite standard deviation is <=0.5 absolute points, i.e. calibration noise is small relative to the 1.5-point acceptance threshold. H0: the spread exceeds 0.5 points, in which case a single-seed comparison is not interpretable.
Boundary conditions. Too few calibration samples (order of a hundred sequences) makes outlier-channel estimation unstable; the ESTIMATE is that per-channel scale estimates need on the order of 128-512 sequences of 2k tokens before seed-to-seed spread falls under 0.5 points, derived from the fact that outlier channels fire on a small fraction of tokens so the effective sample size per channel is far below the nominal token count. Excessively domain-narrow calibration produces a checkpoint that looks excellent in-domain and degrades sharply out-of-domain. Test out-of-domain explicitly.
Evidence. Calibration corpus hash and sample manifest; per-seed checkpoint hashes; per-seed eval scores; a near-duplicate overlap report against eval sets.
Rollback gate. Reject the quantized build if calibration/eval overlap is detected at all, or if seed spread exceeds the acceptance threshold, regardless of how good the best seed looks.""",
 "rewrite", {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.74,
 ["Unfrozen calibration data makes results irreproducible",
  "Calibration/eval overlap inflates measured quality",
  "Single-seed reporting hides calibration variance"],
 ["Calibration corpus content hash and manifest", "Per-seed checkpoint hashes and scores", "Near-duplicate overlap report vs eval sets"]),

("Weight-only quantization must be compared against the alternatives it is competing with, not only against BF16, or the decision is made on a false binary.",
 """Mechanism. The stated goal is lower serving cost. Weight-only quantization is one lever among several with overlapping and sometimes conflicting effects: FP8 weights+activations, KV-cache quantization, speculative decoding, prefix caching, better batching/scheduling, and simply choosing a smaller model. Some compose (weight quant + prefix caching); others contend for the same bottleneck (weight quant + speculative decoding both target decode).
Comparison design. Build a factorial or at least one-factor-at-a-time matrix over {BF16, FP8, INT8-w-only, INT4-w-only} x {KV FP16, KV FP8} x {spec-decode off/on}, all at matched SLO and matched quality gate. Report cost per 1M tokens for each cell. A single-cell A/B cannot support 'quantization is the right cost lever'.
Falsifiable hypothesis. H1: INT4 weight-only is on the cost-quality Pareto frontier, i.e. no other configuration achieves both lower cost and equal-or-better quality. H0: at least one alternative dominates it.
Interaction to watch. Speculative decoding raises arithmetic intensity per step, shifting decode toward compute-bound; the ESTIMATE is that the marginal benefit of weight-only quantization shrinks materially once acceptance-rate-weighted batch effectively multiplies, derived from the roofline argument that bandwidth savings only pay when the kernel is bandwidth-bound.
Evidence. Full cost matrix with SLO attainment per cell; quality gate result per cell; roofline or achieved-bandwidth measurement showing which regime each cell is in.
Rollback gate. Do not ship weight-only quantization if a dominated cell exists; revert if the interaction with an already-deployed decode optimisation erases the win in production traffic.""",
 "rewrite", {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.71,
 ["Comparing only against BF16 ignores dominating alternatives",
  "Weight quantization and speculative decoding contend for the same bottleneck",
  "Single-cell A/B cannot justify a platform-wide decision"],
 ["Cost matrix across configuration cells", "Per-cell quality gate results", "Achieved-bandwidth/roofline regime measurement"]),

("Aggregate benchmark parity can hide localized behavioural regressions, so the quality arm needs distribution-level and behaviour-level probes, not just scores.",
 """Mechanism. Quantization error is not uniform across inputs. It concentrates where activations have heavy-tailed outlier channels and where the correct answer depends on small logit margins. Aggregate accuracy can be flat while long-output stability, format adherence, tool-call JSON validity, refusal behaviour, and non-English performance all degrade.
Probe set. In addition to the score suite, measure: (a) KL divergence / top-1 agreement between BF16 and quantized logits on a held-out prompt set, (b) structured-output validity rate (JSON/tool-call schema pass rate), (c) long-generation degeneration rate (repetition loops at 2k+ output tokens), (d) safety/refusal probe agreement, (e) per-language and per-domain breakdown.
Falsifiable hypothesis. H1: top-1 token agreement with BF16 under greedy decoding is >=97% on the held-out probe set and structured-output validity does not drop by more than 0.5 points. H0: either threshold is missed.
Why this matters operationally. Tool-calling pipelines fail hard, not softly: a 1% drop in JSON validity becomes a 1% hard error rate for downstream automation, which is far more visible to users than a 0.5-point MMLU move. For an agent loop of 10 sequential tool calls the ESTIMATE of end-to-end task failure from that 1% per-call rate is ~9.6%, derived from 1-(0.99^10).
Evidence. Token-agreement and KL histograms; structured-output validity counts with raw failures retained; long-generation samples; per-slice score table.
Rollback gate. Revert on any hard-failure-mode regression (JSON validity, refusal flips, degeneration) even if aggregate scores are unchanged; aggregate parity is necessary but not sufficient.""",
 "rewrite", {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.75,
 ["Aggregate score parity can mask hard failure modes such as invalid tool-call JSON",
  "Outlier-channel sensitivity makes degradation input-dependent",
  "Non-English and long-output regressions are usually untested"],
 ["Top-1 agreement and KL histograms vs BF16", "Structured-output validity counts with raw failures", "Per-language/per-domain score slices"]),

("The experiment must be staged as a guarded rollout with pre-defined abort conditions, because an offline win does not transfer to production traffic automatically.",
 """Mechanism. Offline benchmarks fix the load shape; production does not. The quantized build's advantage is concentrated in decode-bound regimes, so its benefit varies with the hourly mix of long-prompt and long-output requests. Regressions therefore appear as SLO violations at specific traffic phases rather than as a uniform shift.
Rollout design. Stage 1: shadow traffic, quantized replica receives mirrored requests, responses discarded, compare latency and output agreement offline. Stage 2: 5% canary with per-request routing by hash, run at least one full weekly traffic cycle. Stage 3: 50%. Stage 4: full. Each stage has an explicit hold time and a pre-registered metric set.
Falsifiable hypothesis. H1: during the 5% canary over a full weekly cycle, quantized replicas show equal-or-better p95 TTFT and TPOT and no statistically significant increase in downstream task-failure rate. H0: either degrades.
Abort conditions (pre-defined, automated). p95 TTFT +20% versus control, error rate +0.1 absolute points, tool-call validity -0.5 points, or any OOM/kernel fallback event. Abort is automatic and does not require a human decision at 3 a.m. At 5% canary share the ESTIMATE of time needed to detect a +0.1-point error-rate shift is on the order of a day of traffic, derived from needing tens of thousands of canary requests to resolve a difference of that size; set the stage hold time from that calculation rather than from convenience.
Confounders. Canary and control must run on identical hardware SKUs and identical engine versions, and traffic routing must be hash-stable so the same users do not oscillate between arms.
Evidence. Shadow-mode output agreement report; per-stage dashboards with control comparison; automated abort rule configuration committed to version control; incident log.
Rollback gate. One-command revert to the BF16 build with the previous weights kept warm on disk; rollback must be tested in staging before Stage 2 begins.""",
 "rewrite", {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.76,
 ["Offline wins may not transfer to production traffic mixes",
  "Manual abort decisions are unreliable during off-hours incidents",
  "Unstable routing lets the same user oscillate between arms"],
 ["Shadow-mode output agreement report", "Committed automated abort-rule configuration", "Tested rollback procedure record from staging"]),

("Reproducibility is the deliverable: the comparison should be shipped as a re-runnable artefact whose numbers can be regenerated bit-for-bit by a third party.",
 """Mechanism. A quantization comparison has an unusually long chain of hidden state: checkpoint hash, quantization config, engine commit, CUDA/driver version, kernel library version, GPU SKU, clock policy, batching config, and sampling seed. Any of these can move a decode latency number by double digits, so a result reported as a bare percentage is not verifiable.
Artefact requirements. Ship a repository containing: pinned container image digest, exact engine commit SHA, both checkpoint hashes, quantization config file, the load generator with its trace file and hash, the analysis notebook, and a single command that regenerates every figure. Record nvidia-smi output including driver, persistence mode, and clock limits for both arms.
Falsifiable hypothesis. H1: an independent operator re-running the pinned artefact on the same GPU SKU reproduces every headline latency number within 5% and every quality number within 0.3 absolute points. H0: reproduction falls outside those bands, in which case the reported effect is not separable from environment noise.
Boundary conditions. Locked clocks are required for latency reproducibility; with default auto-boost the ESTIMATE of run-to-run variation is on the order of several percent, derived from clock-frequency drift under thermal load, which is the same magnitude as small quantization effects.
Evidence. Container image digest, driver/clock logs, both checkpoint hashes, trace hash, one-command regeneration script, and an independent third-party reproduction report.
Rollback gate. Treat any result that fails independent reproduction as unproven and do not promote the quantized build on its basis; re-run under locked clocks before re-deciding.""",
 "rewrite", {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}, 0.73,
 ["Unpinned environment makes latency deltas unattributable",
  "Auto-boost clock drift is the same magnitude as small quantization effects",
  "Results without a regeneration path cannot be audited"],
 ["Pinned container image digest and driver/clock logs", "Both checkpoint hashes and quantization config", "Independent third-party reproduction report"]),
]

assert len(BODIES) == len(rows) == 10

out = []
for r, (stance, body, decision, qd, conf, risks, ev) in zip(rows, BODIES):
    m = {x["role"]: x["content"] for x in r["messages"]}
    out.append({
        "source_id": r["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": decision,
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": STANCE + stance + "\n\n" + body,
        "quality_dimensions": qd,
        "risks": risks,
        "evidence_required": ev,
        "confidence": conf,
    })

with open(OUT, "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(out))
