P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s = open(P).read()
anchor = "## Run 2026-08-18 - train-batch-0200.jsonl"
entry = """## Run 2026-08-18 - train-batch-0201.jsonl

- Batch file: `results/train-batch-0201.jsonl`
- Corpus slice: `research/ai-infra-expert/corpus/train.jsonl` positional rows 2000-2009 (0-indexed), 10 records
- Source IDs: corpus-02209, corpus-02210, corpus-02211, corpus-02212, corpus-02213, corpus-02214, corpus-02215, corpus-02216, corpus-02217, corpus-02219 (positional slicing; corpus-02218 is absent from the corpus, IDs are not consecutive)
- Progress: 2010/2500 train (stage target 2500, set by the user on 2026-08-18, superseding the original 6000-record full-corpus plan). No validation records are produced in this stage (validation target 0).
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: PASS on first run (`scripts/tb_verify_batch_0201.py`, derived from the batch-0200 verifier via sed rather than rewritten). Checks: per-line JSON parse, batch count 10, all 12 required fields, teacher_lane/teacher_model/calibration_status/decision value constraints, exact character equality of source_user and source_assistant against the original corpus, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness across all 201 batches, and the aggregated 2010-record train sequence being a strict prefix of train.jsonl.
- Repairs: none required this run.
- Final schema check: PASS (TOTAL 2010, VERIFY_PASS).
- Manifest: `MANIFEST.sha256` regenerated over every file in the experiment directory except itself (excluding `__pycache__`); `sha256sum -c` verified all entries.

### Technical topics covered by this batch

All ten source records are near-identical variants of the same prompt: choosing between tensor and pipeline parallelism for a latency-sensitive multi-GPU service, with an explicit falsifiable hypothesis and controlled experiment. Because the batch is homogeneous, each corrected answer is written under a distinct, mutually exclusive analytical stance so the ten records are not paraphrases of one another; stances are also disjoint from those used in batch 0200. This batch's stances: scheduler interaction and chunked prefill as a confounder; warmup, autotune and arm-ordering effects; statistical power and tail-quantile estimation; request-trace realism and output-length distribution; cost per SLO-satisfied request as the real objective; engine and version pinning; per-rank straggler and synchronisation tail; rollout mechanics and canary routing; observability instrumentation overhead asymmetry; and decision reversibility and lock-in.

Every answer carries the shared substrate stating the mechanism (tensor parallelism pays a collective on the critical path of every layer and lowers single-request latency; pipeline parallelism splits by depth, is a throughput mechanism, and gives no single-request latency benefit), the boundary conditions (interconnect class within a parallel group, whether the model fits in memory, offered concurrency, and whether the SLO is denominated in TTFT, TPOT or end-to-end latency), the assumptions that must be stated before any recommendation, the controlled-experiment design with an A/A noise-floor control, the evidence required, the confounders, and pre-committed rollback gates. Every numeric statement is explicitly labelled ESTIMATE with its derivation inline, and quantities that can only come from measurement are declared as MEASURED-pending unknowns rather than asserted.

- Status: PROVISIONAL. These are teacher-B second-opinion rewrites produced by the current conversation model under blind conditions; teacher-A outputs were not read at any point during generation. They are not expert gold labels, have not been human-verified, and say nothing about any model's domain capability.

"""
assert anchor in s
s = s.replace(anchor, entry + anchor, 1)
open(P, "w").write(s)
print("OK")
