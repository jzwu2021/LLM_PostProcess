import io
P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
entry = """## Run 0225 (train-batch-0225.jsonl)

- Batch file: results/train-batch-0225.jsonl
- Corpus range: train.jsonl positional slice [2240:2250]
- Source IDs: corpus-02468 .. corpus-02477 (10 consecutive, corpus order preserved)
- Progress: 2250/2500 train (90.0%); remaining 250. Validation target is 0 by user instruction; no validation-batch file exists or was created.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run of scripts/tb_verify_batch_0225.py (JSONL line-parse, count=10, all 12 required fields, teacher_lane/teacher_model/calibration_status/decision values, byte-exact source_user and source_assistant against corpus, non-empty corrected_answer distinct from source_assistant, ESTIMATE label present, stance marker present, quality_dimensions integer 1-5, non-empty risks and evidence_required, confidence in [0,1], global source_id uniqueness, aggregated sequence is a strict prefix of train.jsonl, zero validation-batch files).
- Repairs: one repair, in the generator script only, before any output was written. The generator source had two malformed triple-quoted string literals (a stray `",\\n  \"\"\"` sequence terminating the `body` field of stances 255 and 258). This was a Python source syntax defect, not a data defect; it was fixed in scripts/tb_gen_batch_0225.py and the batch was then generated once. No batch file was rewritten, no prior batch was touched, and the original corpus was not modified.
- Ad-hoc stance-uniqueness check (scoped): the 10 stance identifiers in this batch (250-259) are unique within the batch and collide with zero stance identifiers across all prior batches. This assertion is deliberately scoped to "unique within this batch and zero collision with history"; it is not a claim that all historical stance headers are globally unique, since early batches predate the stance-header convention.
- Final schema check: VERIFY_PASS, TOTAL 2250.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except MANIFEST.sha256 itself; `sha256sum -c` reports all entries OK.

Technical topics covered by this batch. All ten items are variants of the same prompt family - choosing between tensor and pipeline parallelism for a latency-sensitive multi-GPU service, with an explicit falsifiable hypothesis and a controlled experiment. Each corpus item's assistant turn is a grading rubric rather than an answer, so every item is a rewrite. The ten rewrites are differentiated by ten distinct analytical stances, each attacking the comparison from a different confounder:

- Stance 250: warmup, kernel autotuning, graph capture and cold caches contaminate early measurements asymmetrically across layouts; steady-state onset must be located per arm before comparison.
- Stance 251: collective-library version, topology autodetection and tuning environment variables are silent confounders; an unpinned software environment can move TPOT by more than the inter-layout margin.
- Stance 252: grouped-query and multi-query attention cap the useful tensor-parallel degree, because KV heads replicate rather than shard once TP exceeds the KV-head count, so per-GPU KV savings floor while collective cost keeps rising.
- Stance 253: long-context traffic shifts dominance from the fixed per-token collective term to memory-bound attention over a growing KV cache, so the TP-versus-PP margin is a function of the context-length distribution.
- Stance 254: streaming APIs place proxy buffering, TLS framing and socket coalescing on the same critical path as the token, so client-observed inter-token gaps require server-side per-token timestamps before attribution.
- Stance 255: checkpoint shard layout and the GPUDirect Storage read path set cold-start time, which is recovery time and therefore an availability property, not a steady-state latency property; direct-storage fallback to buffered I/O is silent.
- Stance 256: the layout fixes the indivisible replica size and therefore the autoscaling quantum, topological placement difficulty and stranded-GPU count, so fleet-level SLO violation can invert the single-replica latency ranking.
- Stance 257: profiler and tracing overhead scales with instrumented events per token and is therefore asymmetric across layouts; headline numbers must come from an unprofiled pass, with profiled runs used only for attribution.
- Stance 258: both axes are synchronous, so a single throttled device, a degraded link that fell back in lane width or speed, or an unbalanced PCIe topology sets replica latency; wide TP amplifies straggler exposure because it synchronises twice per layer per token.
- Stance 259: the correct deliverable is a scoped, expiring decision record whose validity is the intersection of every conditioning variable above, with one monitored signal and threshold per variable and an explicit expiry.

Every quantitative claim in this batch is labelled ESTIMATE with its derivation. No value is labelled MEASURED, because no benchmark was executed for this review. These outputs are provisional teacher-B review material produced blind to the teacher-A lane. They are not expert gold, they have not been adjudicated against teacher-A, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step and was not performed here.

"""
s = io.open(P, encoding="utf-8").read()
lines = s.split("\n")
# insert after the leading title block: find first line starting with '## '
idx = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
new = "\n".join(lines[:idx]) + "\n" + entry + "\n".join(lines[idx:])
io.open(P, "w", encoding="utf-8").write(new)
print("MD_UPDATED insert_at_line", idx)
