import io
P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
entry = """## Run 0229 (train-batch-0228.jsonl)

- Batch file: results/train-batch-0228.jsonl
- Corpus range: train.jsonl positional slice [2270:2280]
- Source IDs: corpus-02501 through corpus-02510 (10 items in corpus order; taken by positional slice, not by ID arithmetic, and corpus order is preserved exactly).
- Progress: 2280/2500 train (91.2%); remaining 220. Validation target is 0 by user instruction; no validation-batch file exists or was created.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run of scripts/tb_verify_batch_0228.py (JSONL line-parse, count=10, all 12 required fields, teacher_lane/teacher_model/calibration_status/decision values, byte-exact source_user and source_assistant against corpus, non-empty corrected_answer distinct from source_assistant, ESTIMATE label present, stance marker present, quality_dimensions integer 1-5, non-empty risks and evidence_required, confidence in [0,1], global source_id uniqueness, aggregated sequence is a strict prefix of train.jsonl, zero validation-batch files).
- Repairs: none. The generator ran once and produced the batch; the verifier passed on its first execution. No batch file was rewritten, no prior batch was touched, and neither the original corpus nor any teacher-A artifact was read or modified.
- Final schema check: VERIFY_PASS, TOTAL 2280.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except MANIFEST.sha256 itself; `sha256sum -c` reports all entries OK.

Technical topics covered by this batch. All ten items are variants of the same prompt family - an agent that repeatedly calls a calculator when the answer is already known, requiring metrics plus an intervention, with an explicit falsifiable hypothesis and a controlled experiment. Each corpus item's assistant turn is a grading rubric rather than an answer, so every item is a rewrite. The shared frame separates the three distinct causes of redundant tool calls (instruction-driven, policy-driven, attention-driven), defines the minimum metric set (unnecessary-call rate, tool success, final correctness, trajectory length, tool latency, recovery), and orders interventions by reversibility: prompt/tool-description fix, explicit no-tool action, result cache, then preference or reward training. The ten rewrites are differentiated by ten distinct analytical stances:

- Stance 280: the redundancy judge is itself an unvalidated instrument; replay-based counterfactual labels must score the judge before any intervention is optimised against it.
- Stance 281: tool descriptions that instruct unconditional verification make the behaviour compliant, not pathological; the prompt fix is first because it is the only fully reversible one.
- Stance 282: a result cache makes literal duplicate calls nearly free and yields judge-free ground truth via hit rate, but is blind to semantic redundancy and risks staleness.
- Stance 283: an explicit abstention action puts answer-directly on equal footing with the tools; paired tool-required and no-tool sets separate improved discrimination from a mere threshold shift.
- Stance 284: preference training generalises, but only correctness-matched pairs prevent the objective from expressing "be wrong faster"; a scalar call penalty has an exploitable exchange rate.
- Stance 285: call counts must be converted to end-to-end p95; for a cheap local tool the extra model turn, not the tool round trip, is the dominant latency term.
- Stance 286: post-error retry and redundant calling are the same decision under different context, so anti-redundancy pressure destroys recovery unless fault injection gates it.
- Stance 287: redundancy concentrates by task category; pooled means describe no real segment and post-hoc slicing manufactures wins, so segmentation must be pre-registered.
- Stance 288: trajectory length under optimisation pressure rewards early abandonment identically to efficient completion; length must be conditioned on correctness with a separate abandonment metric.
- Stance 289: a rollback gate presupposes a demonstrated revert; per-segment routing with a warm previous checkpoint bounds exposure, and the drill artifact - not the design intent - authorises rollout.

Every quantitative claim in this batch is labelled ESTIMATE with its derivation. No value is labelled MEASURED, because no benchmark was executed for this review. These outputs are provisional teacher-B review material produced blind to the teacher-A lane. They are not expert gold, they have not been adjudicated against teacher-A, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step and was not performed here.

"""
s = io.open(P, encoding="utf-8").read()
lines = s.split("\n")
idx = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
new = "\n".join(lines[:idx]) + "\n" + entry + "\n".join(lines[idx:])
io.open(P, "w", encoding="utf-8").write(new)
print("MD_UPDATED insert_at_line", idx)
