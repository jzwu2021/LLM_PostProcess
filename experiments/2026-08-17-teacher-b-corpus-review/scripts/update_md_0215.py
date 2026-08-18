P="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(P).read()
HDR="# Experiment: teacher-B corpus review (blind, independent second opinion)\n\n"
assert s.startswith(HDR)
NEW = """## Run 0215 (train-batch-0215.jsonl)

- Batch file: results/train-batch-0215.jsonl
- Corpus range: positional rows 2141-2150 of research/ai-infra-expert/corpus/train.jsonl
- Source IDs: corpus-02358, corpus-02359, corpus-02360, corpus-02361, corpus-02362, corpus-02363, corpus-02364, corpus-02365, corpus-02367, corpus-02369 (slicing is positional, never ID arithmetic; the corpus ID sequence is non-consecutive here - corpus-02366 and corpus-02368 do not exist in the file)
- Progress: 2150/2500 train (350 remaining). Validation target is 0; no validation-batch files exist or were created.
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: FAIL on first attempt. The first draft of the batch used a free-form answer body that did not begin with the "Analytical stance under test:" marker the verifier enforces, so all 10 records failed the stance-marker assertion.
- Repairs performed: the batch generator was rewritten to emit the established structure - a per-item mutually exclusive analytical stance header, the shared assumptions/mechanism/boundary-condition frame, the stance-specific hypothesis-experiment-rollback body, and the source-item critique - and the batch file was regenerated. No corpus file, no previously committed batch, and no teacher-A artifact was touched. The verifier for this run was derived from scripts/tb_verify_batch_0214.py by a single sed substitution of the batch filename; it recomputes the expected total and prefix from the aggregate results directory, so no hard-coded offsets needed editing this time.
- Final schema check: VERIFY_PASS, TOTAL 2150 - per-line JSONL parse, 10 records, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values correct, byte-exact source_user and source_assistant against the corpus, non-empty corrected_answer, confidence within [0,1], globally unique source_id, and the aggregate train sequence a strict prefix of train.jsonl
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself (__pycache__ excluded); sha256sum -c reported all files OK

Technical topics covered by this batch: all ten items are near-homogeneous tensor-versus-pipeline parallelism scenario variants (158-169), so each answer carries a mutually exclusive analytical stance (Stance 150-159) disjoint from every stance used in prior batches, layered on the shared assumption/mechanism/boundary-condition frame. Stances cover correlated power draw and clock capping penalising the tensor arm because collectives synchronise compute bursts across ranks; scheduler admission and preemption policy producing the tail so that unmatched KV admission depth measures the scheduler rather than the layout; tokenizer and output-length asymmetry manufacturing a result by letting the arms do unequal amounts of work; warmup, CUDA-graph capture and autotuning being one-time costs that invert the ranking when charged to steady state; MLP intermediate-width and head-count divisibility setting a hard feasibility ceiling on the tensor degree before any latency argument applies; equal-cost comparison being denominated in GPU-hours per served token rather than GPU count, with a concurrency-dependent crossing point; observability needing to decompose the delta into collective, compute, bubble and queue time or the result cannot generalise beyond the tested version; multi-tenancy and placement acting as a straggler amplifier because an all-reduce completes at its slowest participant while a pipeline stall is locally absorbed; numerical parity from changed reduction order having to be demonstrated before latency is discussed; and recording the outcome as a versioned, canaryable configuration with a declared expiry rather than a permanent architectural verdict.

Status caveat: this output is PROVISIONAL teacher-B review material produced blind, without any visibility into the teacher-A calibration lane. It is not expert gold, has not been human-adjudicated, and is not evidence about any model's domain capability. Every quantitative claim in this batch is labelled ESTIMATE with its derivation stated; no value is MEASURED, because no benchmark run was executed for this review.

"""
open(P,"w").write(HDR+NEW+s[len(HDR):])
print("MD_UPDATED")
