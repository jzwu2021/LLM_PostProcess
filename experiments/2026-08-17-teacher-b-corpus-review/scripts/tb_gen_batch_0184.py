import ast, json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
OUT = f"{EXP}/results/train-batch-0184.jsonl"
START = 1830
N = 10

# Reuse the frozen shared substrate verbatim from batch 0182's generator without executing it.
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
    ("Analytical stance under test: baseline-legitimacy-first - most of the reported win is usually an untuned baseline.",
     "Falsifiable hypothesis H11: at least half of the naive BF16-vs-INT4 throughput delta disappears once the BF16 arm is tuned to its own frontier (KV block count, max_num_seqs, chunked-prefill, scheduler policy). If tuning the baseline does not move it, the baseline was already optimal and that fact must be demonstrated, not assumed.",
     "Run a baseline-tuning sweep as a funded stage of the study, before quantization is measured at all, and publish the BF16 latency-throughput frontier before and after tuning. The common failure is that the BF16 arm ships with defaults while the quantized arm receives weeks of attention, so the comparison measures engineering effort rather than numeric format. Record effort symmetry explicitly: hours spent tuning each arm, number of configurations swept per arm, and the sweep grid, and require the grids to be identical in shape. Then report the quantization delta twice - against the untuned and against the tuned baseline - and treat the tuned-baseline number as the only one admissible for a shipping decision. If the two differ by more than the indifference zone, the headline claim from the untuned comparison is retracted in the same document rather than left circulating."),

    ("Analytical stance under test: calibration-data-governance-first - the calibration set is an uncontrolled input that silently determines the result.",
     "Falsifiable hypothesis H12: re-running the identical quantization recipe with three independently drawn calibration sets of the same size produces per-slice quality deltas whose spread is smaller than the claimed quality budget. If calibration-set choice moves quality by more than the budget, the recipe is not reproducible and no single-run quality number is admissible.",
     "Treat calibration data as a first-class experimental variable with its own hash, provenance record, licence status and domain composition. Draw at least three disjoint calibration sets matched on length and domain mix, quantize three times, and report the between-set variance alongside the point estimate; that variance, not the A/A noise floor alone, is the true error bar on quality. Test the leakage precondition by hashing calibration examples against every eval slice and publishing the collision count, because a zero-collision claim without the hash comparison is an assertion. Record sequence length and sample count sensitivity too, since short calibration sequences systematically under-represent long-context activation ranges and the resulting damage appears only on the long-context slice. The deliverable is a calibration protocol that a different engineer can re-execute per checkpoint release, plus the recurring cost of doing so."),

    ("Analytical stance under test: memory-ledger-first - reconcile every byte of HBM before interpreting any throughput number.",
     "Falsifiable hypothesis H13: the measured HBM high-water mark of each arm equals weights plus activation workspace plus fragmentation plus KV blocks to within a small residual, and the throughput difference between the native-capacity and clamped-KV arms is fully explained by the KV block count difference. If the ledger does not close, an unaccounted allocation is confounding the comparison.",
     "Build the ledger before the benchmark: measured weight bytes on device, engine and activation workspace, CUDA graph and allocator overhead, fragmentation, and KV blocks as the residual, each read from device telemetry rather than computed from a formula. Then verify that the freed bytes actually became KV blocks and not allocator slack, since some engines round capacity to block granularity and quietly discard the remainder. The clamped-KV arm exists precisely to make this separable: equal block counts isolate the bandwidth effect, and the native arm's excess over it is the capacity effect. Publish both numbers with the ledger attached. A study that reports one blended speedup without the ledger cannot distinguish a kernel improvement from a batching improvement, and those two have completely different implications for whether the win survives a workload shift."),

    ("Analytical stance under test: workload-representativeness-first - the benchmark trace decides the answer more than the kernel does.",
     "Falsifiable hypothesis H14: replaying a sampled production trace with its real arrival process, prompt-length and output-length distributions, and prefix-cache hit rate reproduces the synthetic-benchmark speedup to within the noise floor. If the trace-replay gain is materially smaller, the synthetic benchmark over-weighted decode and the shipping case is void.",
     "Derive the workload from telemetry, not from a round-number convention. Extract the joint distribution of prompt and output lengths, the arrival process including burstiness, the share of requests hitting a shared prefix, and the decode-token fraction of total GPU time; that last quantity is the Amdahl denominator and it bounds every claim in the study. Replay the trace against both arms at matched p95 SLO and report the gain under trace replay as the headline, with the synthetic sweep demoted to a mechanism-explaining appendix. Also report gain sensitivity across time-of-day and weekday-versus-weekend slices, because a workload whose decode fraction swings materially across the week has no single speedup. State explicitly which production segments the conclusion does and does not cover, so a later regression on an uncovered segment is a known gap rather than a surprise."),

    ("Analytical stance under test: numerical-correctness-first - verify the quantized graph is arithmetically what you think it is before benchmarking it.",
     "Falsifiable hypothesis H15: the deployed quantized graph reproduces a reference dequantize-then-BF16-GEMM computation to within a stated per-layer tolerance on fixed inputs, and no layer silently falls back to BF16 or to an unintended bit width. If the reference check fails on any layer, all downstream throughput and quality numbers are withdrawn until it passes.",
     "Do arithmetic verification before performance work. On fixed inputs and a fixed seed, compare per-layer outputs of the served graph against a reference implementation and record the relative L2 and max absolute deviation per layer against a pre-declared tolerance. Independently dump the on-device dtype, group size and scale layout for every quantized tensor and diff it against the intended recipe, since partial quantization - embeddings, LM head, or router left in BF16 by a library default - changes both the memory ledger and the quality story while remaining invisible in the config file. Confirm determinism by running the check twice and requiring bit-identical outputs, so later quality deltas are attributable to precision rather than to nondeterministic kernels. Only after this passes is a throughput number meaningful; benchmarking an unverified graph measures an artifact of unknown identity."),

    ("Analytical stance under test: long-context-and-attention-first - weight quantization does nothing for the term that actually grows.",
     "Falsifiable hypothesis H16: as context length grows, the WOQ decode speedup decays toward unity because attention and KV traffic - untouched by weight quantization - come to dominate per-token bytes moved. If measured speedup is flat across context lengths, the sweep is not actually varying resident context or the measurement is wrong.",
     "Sweep context length across at least four points spanning the production distribution's tail and plot speedup against it; the expected shape is monotone decay, and observing it is a positive control that the experiment is measuring what it claims. Decompose per-token bytes moved into weight reads and KV reads at each point so the crossover context length - beyond which WOQ is economically irrelevant - is an explicit deliverable. This is the point where the honest recommendation often becomes KV-cache quantization, grouped-query attention, or paged and offloaded KV instead, each of which is a separate experiment with its own quality risk and must not be bundled into the weight-quantization arm. Report the fraction of production traffic sitting past the crossover, because that fraction directly discounts the annualised saving and is frequently the number that turns a ship decision into a no-ship decision."),

    ("Analytical stance under test: maintenance-burden-first - price the second numeric path over its whole lifetime, not at launch.",
     "Falsifiable hypothesis H17: the annualised saving from WOQ exceeds the annualised recurring cost of maintaining a second numeric path - per-release requantization, re-qualification, dual-stack CI, doubled incident surface and doubled artifact storage - by a margin larger than the indifference zone. If it does not, the correct decision is no-ship even when the benchmark shows a clean win.",
     "Enumerate the recurring costs in engineer-days per release and multiply by the real release cadence: conversion pipeline runtime and GPU cost, calibration data refresh and governance review, the full quality re-qualification suite, canary time, dual-stack CI capacity, and the expected incident cost of on-call engineers reasoning about two numeric paths under pressure. Add the option cost of pinning to a library version whose quantization kernels are stable, which delays unrelated upgrades. Then compare against the alternatives priced in the same unit. Require a named owner and a written deprecation trigger - the conditions under which the quantized path is retired - before launch, because unowned second paths decay silently and are discovered during an incident. The output is an annualised net figure with its assumptions listed, not a benchmark ratio."),

    ("Analytical stance under test: capacity-realisation-first - a speedup that does not remove a replica removes no cost.",
     "Falsifiable hypothesis H18: the measured throughput gain translates into an integer reduction in provisioned replicas at the unchanged headroom and failure-domain policy, verified by matched-window fleet accounting after rollout. If replica count is unchanged, the realised saving is zero regardless of the benchmark ratio.",
     "Model capacity as integers from the start. Compute required replicas as peak demand divided by per-replica capacity at the p95 SLO, then apply the existing headroom multiplier, failure-domain redundancy and autoscaling floor, and check whether the improved per-replica capacity crosses an integer boundary; often it does not, and a 20 percent kernel win yields nothing until it reaches the next boundary. Verify realisation after rollout with matched-window fleet GPU-hours and served tokens rather than with the benchmark ratio, and pre-commit to that reconciliation as a gate on declaring success. Also check that the freed capacity is not immediately consumed by latency-target tightening or by demand induced elsewhere, which is how paper savings evaporate. The deliverable is a before-and-after replica count with the policy constants stated, signed by whoever owns the capacity budget."),

    ("Analytical stance under test: incident-forensics-first - design so that a future outage can be attributed or exonerated quickly.",
     "Falsifiable hypothesis H19: during a rehearsed incident, an on-call engineer with no prior context can determine within a bounded time whether the quantized path is implicated, using only standard dashboards and the arm manifest. If that determination cannot be made quickly, the second numeric path is an unacceptable operational liability regardless of its cost benefit.",
     "Make numeric path a first-class dimension on every serving dashboard, alert and log line, so latency, error rate, output-length distribution and structured-output compliance are always sliceable by arm. Write the incident runbook before launch: how to identify which arm served a given request, how to flip routing to BF16, the measured time-to-safe from the rehearsal, and the escalation path. Rehearse once with a stopwatch and publish the number, because an unrehearsed rollback is an assumption. Add a standing quality canary on the fixed prompt panel with alarm thresholds, since the quantized failure mode is silent drift that no latency or liveness probe detects. Retain per-arm raw generations for a stated window so a quality complaint weeks later can be adjudicated with evidence rather than with recollection. Operability is part of the cost comparison, not a post-launch chore."),

    ("Analytical stance under test: scope-and-claim-hygiene-first - bound what the result licenses before anyone quotes it.",
     "Falsifiable hypothesis H20: the final claim, rewritten with all its qualifiers explicit, still supports the shipping decision; and every qualifier dropped in the executive summary can be shown not to change the sign. If removing a qualifier flips the decision, that qualifier is load-bearing and must appear in the summary.",
     "Write the claim as a single sentence with its scope attached: this checkpoint hash, this bit width and group size, this engine and library build, this TP layout, this workload trace, this context-length range, this p95 SLO, this hardware generation. Then list what the result explicitly does not license - other checkpoints, other model families, other context regimes, other accelerators, other engine versions - and state that each requires re-qualification. Distinguish the three claim classes throughout: MEASURED quantities with CIs and raw artifacts, ESTIMATE quantities with inline derivations, and ASSUMED inputs with their sources; ban unlabelled numbers from the document. Give the result an expiry tied to the next checkpoint or engine upgrade, whichever comes first, because a stale quantization result quoted after an engine bump is a common source of confidently wrong capacity planning. The purpose is to make the claim quotable without being misquotable."),
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
            "rubric omits the bandwidth-bound mechanism, so a model following it can assert a uniform speedup that is mechanically wrong",
            "rubric does not require isolating the KV-capacity confound, permitting a batching gain to be reported as a quantization gain",
            "no ESTIMATE-versus-MEASURED labelling requirement, so byte-count bounds can be presented as throughput results",
            "no rollback threshold or canary requirement, so an unsafe fleet-wide change is not excluded",
            "no requirement to tune the BF16 baseline, so the comparison can measure engineering effort rather than numeric format",
        ],
        "evidence_required": [
            "per-request raw records: arrival, admission, first-token-to-client, per-token timestamps, output length, terminal status",
            "arm configuration manifests with checkpoint, calibration-set, engine and library hashes showing a bit-width-only symmetric difference",
            "per-layer kernel name and achieved-bandwidth traces for both arms, to exclude silent dequant-GEMM fallback",
            "clamped-KV arm and native-capacity arm results reported separately, with the HBM ledger closing to a small residual",
            "per-slice quality scores with bootstrap 95% CIs on absolute deltas, plus raw generations and sampled side-by-side diffs",
            "BF16-vs-BF16 A/A noise-floor measurement across seeds and physical GPUs",
            "tuned-BF16 frontier with effort-symmetry accounting (sweep grid and hours per arm)",
            "matched-window fleet GPU-hour, token and replica-count accounting confirming the saving was realised",
        ],
        "confidence": 0.62,
    })

with open(OUT, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("WROTE", OUT, len(out))
print("IDS", ",".join(r["source_id"] for r in out))
print("OPENINGS_DISTINCT", len({r["corrected_answer"][:200] for r in out}))
