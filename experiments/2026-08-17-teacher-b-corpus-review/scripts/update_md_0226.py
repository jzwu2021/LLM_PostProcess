import io
P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
entry = """## Run 0227 (train-batch-0226.jsonl)

- Batch file: results/train-batch-0226.jsonl
- Corpus range: train.jsonl positional slice [2250:2260]
- Source IDs: corpus-02479, corpus-02480, corpus-02481, corpus-02483, corpus-02484, corpus-02485, corpus-02486, corpus-02487, corpus-02488, corpus-02489 (10 items in corpus order; note the source IDs are not consecutive - corpus-02482 is absent from train.jsonl - so the batch was taken by positional slice, not by ID arithmetic, and corpus order is preserved exactly).
- Progress: 2260/2500 train (90.4%); remaining 240. Validation target is 0 by user instruction; no validation-batch file exists or was created.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run of scripts/tb_verify_batch_0226.py (JSONL line-parse, count=10, all 12 required fields, teacher_lane/teacher_model/calibration_status/decision values, byte-exact source_user and source_assistant against corpus, non-empty corrected_answer distinct from source_assistant, ESTIMATE label present, stance marker present, quality_dimensions integer 1-5, non-empty risks and evidence_required, confidence in [0,1], global source_id uniqueness, aggregated sequence is a strict prefix of train.jsonl, zero validation-batch files).
- Repairs: none. The generator ran once and produced the batch; the verifier passed on its first execution. No batch file was rewritten, no prior batch was touched, and neither the original corpus nor any teacher-A artifact was read or modified.
- Ad-hoc stance-uniqueness check (scoped): the 10 stance identifiers in this batch (260-269) are unique within the batch and collide with zero stance identifiers across all prior batches. This assertion is deliberately scoped to "unique within this batch and zero collision with history"; it is not a claim that all historical stance headers are globally unique, since early batches predate the stance-header convention.
- Final schema check: VERIFY_PASS, TOTAL 2260.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except MANIFEST.sha256 itself; `sha256sum -c` reports all entries OK.

Technical topics covered by this batch. All ten items are variants of the same prompt family - choosing between tensor and pipeline parallelism for a latency-sensitive multi-GPU service, with an explicit falsifiable hypothesis and a controlled experiment. Each corpus item's assistant turn is a grading rubric rather than an answer, so every item is a rewrite. The ten rewrites are differentiated by ten distinct analytical stances:

- Stance 260: attention-head and KV-group divisibility, not fabric bandwidth, is the first hard ceiling on TP degree; past the KV-group count, KV replicates instead of sharding so per-GPU KV bytes stop falling while two all-reduces per layer keep accruing.
- Stance 261: prefill is compute-bound and decode is memory-bandwidth-bound, so their optimal layouts differ; a fused deployment sacrifices one SLO, and disaggregation is only correct while KV-handoff time stays under the TTFT slack.
- Stance 262: the pipeline bubble fraction is a function of in-flight micro-batches, so a PP verdict is really a verdict about the concurrency distribution, and idle troughs pay the full penalty.
- Stance 263: blast radius and recovery differ per axis - TP synchronises twice per layer so a single throttled device sets whole-group latency, and straggler amplification grows with group size while PP localises to a stage.
- Stance 264: quantization scales down the bandwidth term but not the collective term, so the optimal TP degree shrinks at lower precision; format and layout must never move in the same arm, and every format arm needs a held-out quality check.
- Stance 265: crossing a node boundary is a discrete change in the cost model; inter-node TP pays L*2 slow-path all-reduces while inter-node PP pays (PP-1) point-to-point sends, and a silent RDMA-to-host-staging fallback voids the comparison.
- Stance 266: chunked prefill, maximum batched tokens and prefix-cache warmth set the tokens-per-step that collectives amortise against, so unpinned batching policy is a confounder of the same magnitude as the layout choice.
- Stance 267: aggregate throughput and per-request latency move in opposite directions across the two axes, so the deliverable is a throughput-versus-latency frontier with the SLO ceiling marked, not a scalar.
- Stance 268: memory accounting must be per-rank and at peak, including collective buffers, activation workspaces and allocator fragmentation; sizing on averages produces an out-of-memory or eviction cliff under burst rather than graceful degradation.
- Stance 269: the verdict is conditional on model hash, topology, version manifest, numeric format, batching policy, traffic mix and SLO definition, so it must be a dated decision record with one monitored invalidation trigger per fact and a scheduled re-run designed to falsify it.

Every quantitative claim in this batch is labelled ESTIMATE with its derivation. No value is labelled MEASURED, because no benchmark was executed for this review. These outputs are provisional teacher-B review material produced blind to the teacher-A lane. They are not expert gold, they have not been adjudicated against teacher-A, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step and was not performed here.

"""
s = io.open(P, encoding="utf-8").read()
lines = s.split("\n")
idx = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
new = "\n".join(lines[:idx]) + "\n" + entry + "\n".join(lines[idx:])
io.open(P, "w", encoding="utf-8").write(new)
print("MD_UPDATED insert_at_line", idx)
