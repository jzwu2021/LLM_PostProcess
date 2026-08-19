import re
P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
NEW = """## Run 0232 (train-batch-0231.jsonl)

- Batch file: results/train-batch-0231.jsonl
- Corpus range: train.jsonl positional slice [2300:2310]
- Source IDs: corpus-02534, corpus-02535, corpus-02536, corpus-02537, corpus-02538, corpus-02539, corpus-02540, corpus-02541, corpus-02542, corpus-02543 (10 items in corpus order; taken by positional slice, not by ID arithmetic, and corpus order is preserved exactly).
- Progress: 2310/2500 train (92.4%); remaining 190. Validation target is 0 by user instruction; no validation-batch file exists or was created.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run of scripts/tb_verify_batch_0231.py (JSONL line-parse, count=10, all 12 required fields, teacher_lane/teacher_model/calibration_status/decision values, byte-exact source_user and source_assistant against corpus, non-empty corrected_answer distinct from source_assistant, ESTIMATE label present, stance marker present, quality_dimensions integer 1-5, non-empty risks and evidence_required, confidence in [0,1], global source_id uniqueness, aggregated sequence is a strict prefix of train.jsonl, zero validation-batch files).
- Repairs: none. The stance module scripts/stances_0231.py was authored fresh with ten new stances, and scripts/gen_batch_0231.py plus scripts/tb_verify_batch_0231.py were derived from the previous run's files by sed substitution of the batch number and the positional slice bounds; the derived import line was confirmed to point at stances_0231 before execution. The generator ran once and the verifier passed on its first execution. No batch file was rewritten, no prior batch was touched, and neither the original corpus nor any teacher-A artifact was read or modified.
- Final schema check: VERIFY_PASS, TOTAL 2310.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except MANIFEST.sha256 itself; `sha256sum -c` reports all entries OK.

Technical topics covered by this batch. All ten items continue the same prompt family - an agent that repeatedly calls a calculator when the answer is already known, requiring metrics plus an intervention, with an explicit falsifiable hypothesis and a controlled experiment. Each corpus item's assistant turn is a grading rubric rather than an answer, so every item is a rewrite. The shared frame is unchanged: three distinct causes of redundant tool calls, a minimum metric set, and an intervention ladder ordered by reversibility. This batch shifts the differentiation toward measurement validity, serving-layer bounds and cost accounting:

- Stance 310: the unit of analysis must be the trajectory, not the call, because redundancy is heavy-tailed and concentrates in looping trajectories; the turn cap in force must be reported or the tail is invisible by construction.
- Stance 311: loop detection belongs in the serving layer as a deterministic bound, since a probabilistic policy gives no worst-case guarantee; refusal must return an informative observation rather than silently dropping the call.
- Stance 312: call-signature normalization is where the measurement quietly breaks; the rule must ship as versioned code with audited false-merge and false-split rates, because plausible variants move the rate by as much as the interventions do.
- Stance 313: the tool description is an instruction surface closer to the decision point than the system prompt, and stating an explicit negative precondition there is a revertible schema change that should be tried before any prompt campaign.
- Stance 314: latency accounting must be end-to-end and percentile-based, because a per-call check costs every trajectory uniformly while saving only tail trajectories, so mean improvements can conceal p99 regressions.
- Stance 315: the offline redundancy adjudicator is an unvalidated instrument; it needs inter-rater agreement, a constructed clear-cut bias set, and a pinned judge checkpoint, and effects below the disagreement band are not resolvable.
- Stance 316: apparently redundant calls may follow context eviction, making them context-management failures immune to policy interventions; the diagnostic is a join against per-turn rendered-context logs, and the absence of those logs is itself the finding.
- Stance 317: the cost model must be expressed in currency and utilization, not call counts, since freed capacity has value only when capacity is the binding constraint - which for most deployments means only inside the peak window.
- Stance 318: harness determinism is a precondition, because unpinned batched serving amplifies single-token divergence into whole-branch trajectory differences at a magnitude comparable to the claimed effects; effects below the measured noise floor are null results.
- Stance 319: the closing position is that the source assistant turn is a rubric, and many near-identical variants multiply that defect rather than diluting it; the supervision signal is consistently wrong in a specific direction, and deduplication decisions require measured cluster composition.

Every quantitative claim in this batch is labelled ESTIMATE with its derivation. No value is labelled MEASURED, because no benchmark was executed for this review. These outputs are provisional teacher-B review material produced blind to the teacher-A lane. They are not expert gold, they have not been adjudicated against teacher-A, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step and was not performed here.

"""
s = open(P).read()
marker = "## Run 0231 (train-batch-0230.jsonl)"
assert marker in s
i = s.index(marker)
s = s[:i] + NEW + s[i:]
open(P, "w").write(s)
print("OK")
