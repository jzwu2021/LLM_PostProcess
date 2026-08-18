import ast, json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT = f"{EXP}/results/train-batch-0185.jsonl"
START = 1840
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
    ("Analytical stance under test: arithmetic-intensity-first - decide from the roofline before any kernel is built.",
     "Falsifiable hypothesis H21: the per-token decode speedup from weight-only INT4 is bounded above by the ratio of BF16 to INT4 weight bytes read per token, and the measured speedup at batch size 1 lands within a stated fraction of that bound. If measured speedup exceeds the byte-ratio bound, the two arms are not doing the same work and the comparison is void.",
     "Compute the roofline before benchmarking. Decode at small batch is weight-bandwidth bound: bytes moved per token is approximately the weight footprint plus the KV read, so halving or quartering weight bytes gives an upper bound on speedup that no kernel can beat. ESTIMATE: for a dense model whose BF16 weights occupy W bytes, INT4 group-wise weights occupy roughly W/4 plus scale and zero-point overhead of a few percent at group size 128, so the decode bound is near 3.5x rather than 4x; derivation is bytes-per-token ratio only, assuming perfect bandwidth utilisation and unchanged KV traffic, and it is an ESTIMATE, not MEASURED. As batch size rises, GEMMs become compute bound and the same weight read amortises across more tokens, so the quantization benefit decays toward zero; state the batch size at which your measured curve crosses that knee. Prefill is compute bound from the start, so a weight-only scheme should show near-zero TTFT gain and any observed prefill win is evidence of a confound. Publish the roofline prediction before the run so the measurement can falsify it rather than be narrated around it."),

    ("Analytical stance under test: dequantization-overhead-first - the win is in bytes read but the loss is in the inner loop.",
     "Falsifiable hypothesis H22: the deployed INT4 path executes fused dequant-GEMM kernels for at least a stated fraction of linear-layer FLOPs, and achieved memory bandwidth on those kernels is within a stated fraction of the BF16 arm's. If a material fraction falls back to a materialise-then-GEMM path, the arm is misconfigured and its numbers must be discarded, not caveated.",
     "Weight-only quantization only pays if dequantization is fused into the GEMM epilogue-prologue and never materialises a BF16 weight tile in HBM. Verify this mechanically rather than by trusting the config flag: capture a kernel trace, list the kernel names actually executed per layer, and confirm the fused path covers the layers you assumed. Common silent failures are unsupported shapes, head dimensions or group sizes falling back to a reference path, and fused-MoE or attention-projection layers left in BF16 while only the MLP is quantized. Record the achieved-bandwidth counter per kernel and compare against the arm's device peak; a fused INT4 kernel far below the BF16 arm's achieved bandwidth indicates the dequant is serialising rather than overlapping. Also record which layers were deliberately excluded from quantization, because partial coverage changes both the byte ratio and the quality result and is the single most common reason a reproduction fails to match a published number."),

    ("Analytical stance under test: KV-cache-confound-first - separate the memory freed from the memory used.",
     "Falsifiable hypothesis H23: with KV-cache blocks clamped to the BF16 arm's count, the INT4 arm's throughput gain shrinks by more than half. If clamping does not change the gain, the gain is genuinely from weight bandwidth; if it collapses, most of the reported win was extra concurrency, not quantization.",
     "Smaller weights free HBM, and a serving engine will spend that HBM on more KV blocks, which raises max concurrency, which raises throughput. That is a real benefit but it is a different mechanism from faster weight reads, and conflating them makes the result non-transferable to any deployment with a different context budget. Run both arms twice: once with KV block count clamped identically, isolating the weight-bandwidth effect, and once at each arm's native capacity, measuring the deployable end-to-end effect. Report both numbers with the mechanism named. Close an HBM ledger for each arm - weights, KV blocks, activation and workspace, fragmentation, engine overhead - and require the residual to be small; an unclosed ledger means you do not know what the freed memory was spent on. State the context-length distribution of the workload trace, because the KV-driven share of the win scales with it and a result gathered on short prompts will not hold on long ones."),

    ("Analytical stance under test: quality-measurement-power-first - a quality claim without an error bar is not a claim.",
     "Falsifiable hypothesis H24: the per-slice quality delta between arms is smaller in absolute value than a pre-registered indifference threshold, with bootstrap 95 percent confidence intervals on the absolute delta that exclude threshold violation. If the interval is wider than the threshold, the eval is underpowered and no ship-or-not conclusion is licensed.",
     "Pre-register the slices, the threshold and the sample size before looking at any result. Compute required sample size from a BF16-versus-BF16 A/A run: the A/A spread across seeds and physical GPUs is the noise floor, and any A/B delta inside it is unmeasured, not zero. Report absolute deltas with bootstrap CIs per slice, never a single averaged score, because quantization damage concentrates in specific regimes - long context, structured output and constrained decoding, rare-token and multilingual tails, and long chain-of-thought where small per-token drift compounds. Fix sampling parameters, seeds and stop conditions across arms and record them; a temperature difference between arms invalidates the comparison silently. Retain raw generations for both arms so a disputed quality claim can be adjudicated later with evidence, and sample a fixed panel of side-by-side diffs for human reading, since aggregate scores routinely miss format regressions that break downstream parsers."),

    ("Analytical stance under test: workload-representativeness-first - the trace is the experiment, the harness is a detail.",
     "Falsifiable hypothesis H25: the latency-throughput frontier measured under a replayed production arrival trace ranks the two arms the same way as the closed-loop synthetic benchmark. If the ranking changes, the synthetic result must not be used for a capacity or shipping decision.",
     "Closed-loop benchmarks with fixed concurrency measure a different system than open-loop arrivals with queueing, and quantization interacts with queueing through admission and batch composition. Replay an arrival trace with the production distribution of prompt and output lengths, burstiness and cancellation rate, and report the full latency-throughput frontier rather than a single operating point, because arms often cross. Define the SLO first - which percentile, over what window, measured at the client including queueing - and read throughput at the SLO rather than at saturation. Exclude warmup, compilation and cache-fill from the measurement window explicitly and state the exclusion rule. Record cancelled and truncated requests separately; a system that degrades by dropping work can otherwise appear faster. State the context-length range the result covers and refuse to extrapolate outside it."),

    ("Analytical stance under test: cost-accounting-first - the unit of decision is money per served token, not tokens per second.",
     "Falsifiable hypothesis H26: cost per million served tokens at the p95 SLO improves by more than the pre-registered indifference margin once quantization pipeline cost, re-qualification effort and dual-path operational overhead are amortised over the checkpoint's expected life. If the margin is not cleared, throughput gain is not a reason to ship.",
     "Build the cost model before the benchmark and populate it afterwards. Include the one-off costs the throughput ratio hides: calibration data curation, quantization runs and their failures, per-checkpoint re-qualification, extra eval compute, engineering hours to maintain a second numeric path, and the incident risk of that path. ESTIMATE any figure you cannot measure and label it inline with its derivation; MEASURED figures must carry their artifact reference. Amortise one-off cost over the number of checkpoints the recipe will serve, which is usually small and is the dominant term people omit. Then check the realisation condition: does the improvement remove an integer number of replicas at the unchanged headroom policy, or does it merely raise utilisation headroom that is never harvested. State the decision rule numerically before the data arrives, so the outcome cannot be renegotiated after the fact."),

    ("Analytical stance under test: reproducibility-and-provenance-first - a result nobody can re-run is a rumour.",
     "Falsifiable hypothesis H27: an independent engineer, given only the published manifests and scripts, reproduces the headline throughput and quality numbers within the stated A/A noise floor on fresh hardware. If reproduction lands outside that band, the original result is provisional and cannot be cited.",
     "Freeze and hash every input: checkpoint, quantization recipe with bit width and group size and symmetry, calibration set, engine and library and driver versions, kernel selection, tensor-parallel and pipeline layout, scheduler settings, and the workload trace. Require that arm manifests differ in exactly the intended dimension and diff them mechanically rather than by eye; an unintended engine-version difference between arms is the most common invisible confound. Pin seeds and record physical GPU identity, since per-device clock and thermal variation is a real source of spread. Publish scripts that regenerate every table from raw artifacts, and run the reproduction once yourself on different hardware before publishing. Give the artifact bundle an expiry tied to the next engine or checkpoint bump, because a stale number quoted after an upgrade is a routine cause of confidently wrong planning."),

    ("Analytical stance under test: rollout-safety-first - the experiment's job is to make an unsafe change impossible to ship quietly.",
     "Falsifiable hypothesis H28: a shadow and canary rollout at a stated traffic fraction detects any quality or latency regression exceeding the pre-registered threshold within a bounded observation window, and rollback to the BF16 path completes within a rehearsed and measured time-to-safe. If detection or rollback cannot be demonstrated, the change is not shippable regardless of the benchmark result.",
     "Sequence the rollout as shadow, then canary, then staged ramp, each with a pre-committed abort threshold on latency percentiles, error rate, output-length distribution and structured-output compliance. Quantization damage is typically silent: liveness and latency probes stay green while output quality drifts, so a standing quality canary on a fixed prompt panel with alarm thresholds is mandatory, not optional. Keep the BF16 path warm and routable for the entire ramp and rehearse the flip with a stopwatch, publishing the measured time-to-safe rather than asserting rollback is easy. Make numeric path a slicing dimension on every dashboard, alert and log line so an on-call engineer can implicate or exonerate the quantized arm quickly. Write the abort thresholds into the change record before the ramp begins, because thresholds chosen after seeing the data are not thresholds."),

    ("Analytical stance under test: generalisation-boundary-first - state where the result stops being true.",
     "Falsifiable hypothesis H29: the sign and approximate magnitude of the quantization benefit hold across the deployment's actual context-length range, batch-size range and hardware generation, verified by measuring at the endpoints rather than the midpoint. If the sign flips at any endpoint, the single-point result must not be generalised.",
     "A quantization result is a function of operating regime, not a property of the model. Measure at the extremes of the regimes you intend to cover: shortest and longest context in the trace, smallest and largest realistic batch, and each accelerator generation in the fleet, since bandwidth-to-FLOPs ratios and kernel support differ enough to move the sign. At large batch the weight read amortises and the benefit decays; at long context KV traffic dominates weight traffic and the benefit decays again, so the headline number gathered at short context and small batch is usually the most favourable point in the space and the least representative one. Publish the frontier across regimes rather than a point, and list explicitly what the result does not license: other model families, other bit widths, other engine versions, other accelerators. Each of those requires re-qualification, and saying so in the document is what prevents the number from being misapplied by someone who was not in the room."),

    ("Analytical stance under test: decision-governance-first - fix who decides and on what evidence before the numbers exist.",
     "Falsifiable hypothesis H30: applying the pre-registered decision rule mechanically to the final evidence pack yields the same ship-or-not verdict as the team's narrative recommendation. If the two diverge, the narrative is being driven by something outside the stated evidence and that must be surfaced.",
     "Write the decision rule as executable arithmetic before data collection: the cost-per-token improvement margin required, the per-slice quality indifference threshold, the required reproduction agreement, the required time-to-safe, and the named owner for each. Pre-register the analysis plan so slice selection and stopping cannot drift after results appear, and state what would cause you to abandon quantization entirely rather than only what would cause you to adopt it. Separate the claim classes throughout the write-up: MEASURED with confidence intervals and raw artifacts, ESTIMATE with inline derivation, ASSUMED with source; ban unlabelled numbers. Require the evidence pack to be complete before the review meeting, since a missing arm manifest or absent A/A run is grounds for deferral rather than for a caveat. The deliverable is a one-sentence claim with its scope attached, an expiry date, and a signed reconciliation obligation to check after rollout whether the predicted saving was actually realised."),
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
            "rubric omits the bandwidth-bound decode mechanism, so a model following it can assert a uniform speedup that is mechanically wrong",
            "rubric does not require isolating the KV-capacity confound, permitting a batching gain to be reported as a quantization gain",
            "no requirement to verify fused dequant-GEMM kernel selection, so a silent reference-path fallback can be reported as a quantization result",
            "no ESTIMATE-versus-MEASURED labelling requirement, so byte-count bounds can be presented as measured throughput",
            "no rollback threshold, canary or time-to-safe requirement, so an unsafe fleet-wide numeric-path change is not excluded",
        ],
        "evidence_required": [
            "per-request raw records: arrival, admission, first-token-to-client, per-token timestamps, output length, terminal status",
            "arm configuration manifests with checkpoint, calibration-set, engine, library and driver hashes showing a bit-width-only symmetric difference",
            "per-layer kernel name and achieved-bandwidth traces for both arms, to exclude silent dequant-GEMM fallback and partial quantization coverage",
            "clamped-KV arm and native-capacity arm results reported separately, with the HBM ledger closing to a small residual",
            "per-slice quality scores with bootstrap 95% CIs on absolute deltas, plus retained raw generations and a sampled side-by-side diff panel",
            "BF16-vs-BF16 A/A noise-floor measurement across seeds and physical GPUs",
            "replayed production arrival trace with prompt and output length distributions, plus the full latency-throughput frontier at the stated SLO percentile",
            "rehearsed rollback record with measured time-to-safe, and matched-window fleet GPU-hour, token and replica-count accounting confirming realisation",
        ],
        "confidence": 0.62,
    })

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("WROTE", OUT, len(out))
print("IDS", ",".join(r["source_id"] for r in out))
print("OPENINGS_DISTINCT", len({r["corrected_answer"][:200] for r in out}))
