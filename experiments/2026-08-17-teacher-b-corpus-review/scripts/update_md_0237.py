import io, os
EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
MD = f"{EXP}/EXPERIMENT.md"
TITLE = "# Experiment: teacher-B corpus review (blind, independent second opinion)\n"

entry = """
## Round 0237 - train-batch-0237.jsonl

- Batch file: results/train-batch-0237.jsonl
- Corpus interval (positional): train.jsonl rows [2360, 2370)
- Source IDs: corpus-02602 .. corpus-02611 (10 items, contiguous, corpus order preserved)
- Progress: train 2370/2500 (remaining 130); validation 0/0 by user instruction, no validation batch files exist or were created
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_0237.py, derived from verify_0236.py by sed with the positional window moved to [2360,2370) and the aggregate count moved to 2370). The generator selected message content by role from the start, so no byte-equality, field-set or uniqueness failure occurred.
- Repair actions: none required this round. Original corpus untouched; no previously committed batch was modified; no teacher-A path was read, opened or grepped.
- Final schema check: VERIFY_PASS batch=10 aggregate=2370 prefix=ok ids_unique=ok - JSONL line-parseable and newline-terminated, 10 rows, exactly the 12 required fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, source_user/source_assistant byte-equal to corpus by role, corrected_answer non-empty with unique stance headers within the batch, quality_dimensions three ints in [1,5] with bool excluded, risks/evidence_required non-empty string arrays, confidence float in [0,1], global source_id uniqueness, aggregate 2370 rows a strict prefix of train.jsonl, no validation-batch files.
- Manifest: MANIFEST.sha256 regenerated over all files in this experiment directory except MANIFEST.sha256 itself; sha256sum -c reported all OK.

Technical topics covered by this batch. All ten source items instantiate the same degenerate template - an agent-reliability prompt about an agent that repeatedly calls a calculator when the answer is already known, scenario variants 102 through 111 - each paired with an assistant turn that is a 313-character grading rubric rather than an actual answer, so all ten are rewrites. The ten rewrites take ten distinct, non-overlapping analytical stances, none repeating a stance used in rounds 0234-0236: (v102) four-term end-to-end cost accounting for a redundant call, with the amortized re-prefill term over remaining turns identified as the dominant and routinely omitted term; (v103) prefix-cache interaction, where a system-prompt edit invalidates cached prefixes and produces a p95 spike that is a rollout artifact rather than an intervention regression, with the explicit instruction that auto-rollback must not fire inside the re-warm window; (v104) tool-schema surface as the cheapest reversible lever, with a pre-registered escalation order (tool description, system-prompt exemplar, decoding constraints, preference optimization) in which the irreversible step is not started until the reversible ones are exhausted; (v105) stratification by prompt template, showing that a single global redundancy rate cannot distinguish a policy-level defect from a template-concentrated one and that the distinction is decidable from existing traces; (v106) continuous-batching effects, where freed KV-cache capacity converts into admitted concurrency rather than lower per-request latency, making fleet throughput at fixed p95 the correct signal and preemption/recompute counts the rollback trigger; (v107) sealed held-out evaluation with iteration counting as the bound on selection optimism, and the rule that a burned held-out set must be recut; (v108) the symmetric under-call failure mode, argued to be an earlier and more sensitive indicator than aggregate task success because success metrics dilute a subpopulation regression; (v109) attribution discipline separating runtime/system changes from model changes, with the mechanical rule that a comparison spanning both categories is invalid and must be re-run with the runtime pinned; (v110) preference-optimization preconditions and pair construction, where preferred and rejected trajectories must differ only in the redundant call with final-answer correctness held fixed, plus a restore drill that is actually exercised rather than assumed; (v111) programme close-out, with a hash-pinned claim template written before the first arm runs and diffed against the original proposal to prevent retrospective narrative expansion.

Every rewrite states its assumptions explicitly, gives the token-level mechanism by which tool invocation is a distributional property of the induced policy rather than a patchable controller defect, lists the boundary conditions that flip the recommendation, states a falsifiable hypothesis with its explicit refutation condition, specifies a single-variable controlled experiment on a hash-pinned replay, enumerates the evidence artifacts required to adjudicate it, and defines a rollback gate. No quantitative value in this batch is labelled MEASURED; every non-artifact number is labelled ESTIMATE and carries its derivation, because no run was executed for this review.

Status caveat. These outputs are provisional teacher-B review material produced under blind review - no teacher-A artifact was read, opened or grepped at any point while producing this batch. They are not expert gold, they have not been adjudicated by a human expert, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step outside the scope of this worker.
"""

txt = open(MD).read()
assert txt.startswith(TITLE)
rest = txt[len(TITLE):]
open(MD, "w").write(TITLE + entry + rest)
print("MD_UPDATED")
