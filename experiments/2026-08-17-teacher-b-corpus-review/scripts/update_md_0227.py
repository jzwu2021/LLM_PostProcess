import io
P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
entry = """## Run 0228 (train-batch-0227.jsonl)

- Batch file: results/train-batch-0227.jsonl
- Corpus range: train.jsonl positional slice [2260:2270]
- Source IDs: corpus-02490, corpus-02491, corpus-02492, corpus-02493, corpus-02494, corpus-02495, corpus-02496, corpus-02497, corpus-02499, corpus-02500 (10 items in corpus order; the source IDs are not consecutive - corpus-02498 is absent from train.jsonl - so the batch was taken by positional slice, not by ID arithmetic, and corpus order is preserved exactly).
- Progress: 2270/2500 train (90.8%); remaining 230. Validation target is 0 by user instruction; no validation-batch file exists or was created.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run of scripts/tb_verify_batch_0227.py (JSONL line-parse, count=10, all 12 required fields, teacher_lane/teacher_model/calibration_status/decision values, byte-exact source_user and source_assistant against corpus, non-empty corrected_answer distinct from source_assistant, ESTIMATE label present, stance marker present, quality_dimensions integer 1-5, non-empty risks and evidence_required, confidence in [0,1], global source_id uniqueness, aggregated sequence is a strict prefix of train.jsonl, zero validation-batch files).
- Repairs: none. The generator ran once and produced the batch; the verifier passed on its first execution. No batch file was rewritten, no prior batch was touched, and neither the original corpus nor any teacher-A artifact was read or modified.
- Final schema check: VERIFY_PASS, TOTAL 2270.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except MANIFEST.sha256 itself; `sha256sum -c` reports all entries OK.

Technical topics covered by this batch. All ten items are variants of the same prompt family - choosing between tensor and pipeline parallelism for a latency-sensitive multi-GPU service, with an explicit falsifiable hypothesis and a controlled experiment. Each corpus item's assistant turn is a grading rubric rather than an answer, so every item is a rewrite. The ten rewrites are differentiated by ten distinct analytical stances:

- Stance 270: speculative decoding turns decode's one-token step into a k-token verification step, amortising TP collectives; the gain scales with measured accepted-token count rather than proposal length, and acceptance drift makes a speculation-tuned layout fragile.
- Stance 271: collective algorithm, protocol and chunk-size selection sit between layout and fabric; ring versus tree at decode message sizes can dominate the inter-degree margin, so unpinned collective heuristics confound any TP sweep.
- Stance 272: autotuning, graph capture and peer-connection setup make cold start layout-dependent and longer for wider groups, so including warmup measures startup rather than steady state - yet cold start is real for elastic deployments.
- Stance 273: long context shifts the bottleneck from weight bandwidth to KV bandwidth, creating a context length above which a previously net-negative TP degree becomes optimal; the crossover must be measured against the real prompt-length tail.
- Stance 274: for routed mixture-of-experts models the dominant cost is a load-imbalanced all-to-all, so per-step latency tracks max per-expert token count, and capacity-factor fixes buy latency by dropping tokens.
- Stance 275: multi-tenant colocation makes fabric and memory shared; synchronous per-layer collectives sample the contention tail once per participant, so quiet-node benchmarks understate wide-TP tail latency.
- Stance 276: end-to-end p95 is queue wait plus service time, and near saturation the routing policy moves the tail more than the layout does, so layout comparisons run over a bad router are not interpretable.
- Stance 277: reduction order differs by shard count, so a layout change is a model-behavior change; greedy outputs can diverge, breaking golden tests and output-keyed caches and inviting misdiagnosis during incidents.
- Stance 278: the decision quantity is cost per sustained token at the SLO, because minimum-latency and peak-throughput rankings both select operating points off the cost-optimal point and hide unequal GPU counts.
- Stance 279: layout degree is not a runtime knob, so the rollout mechanism is part of the decision; canarying with a pre-registered auto-revert bounds blast radius, and an unrehearsed revert path makes a bad rollout unbounded.

Every quantitative claim in this batch is labelled ESTIMATE with its derivation. No value is labelled MEASURED, because no benchmark was executed for this review. These outputs are provisional teacher-B review material produced blind to the teacher-A lane. They are not expert gold, they have not been adjudicated against teacher-A, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step and was not performed here.

"""
s = io.open(P, encoding="utf-8").read()
lines = s.split("\n")
idx = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
new = "\n".join(lines[:idx]) + "\n" + entry + "\n".join(lines[idx:])
io.open(P, "w", encoding="utf-8").write(new)
print("MD_UPDATED insert_at_line", idx)
