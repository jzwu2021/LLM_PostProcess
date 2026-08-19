import os
EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
MD = f"{EXP}/EXPERIMENT.md"
TITLE = "# Experiment: teacher-B corpus review (blind, independent second opinion)\n"

entry = """
## Round 0238 - train-batch-0238.jsonl

- Batch file: results/train-batch-0238.jsonl
- Corpus interval (positional): train.jsonl rows [2370, 2380)
- Source IDs: corpus-02612, corpus-02614, corpus-02616 .. corpus-02623 (10 items, corpus order preserved; the corpus itself skips corpus-02613 and corpus-02615, so the ID run is non-consecutive while the positional slice is contiguous)
- Progress: train 2380/2500 (remaining 120); validation 0/0 by user instruction, no validation batch files exist or were created
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_0238.py, derived from verify_0237.py by sed with the positional window moved to [2370,2380) and the aggregate count moved to 2380). The generator selected message content by role, so no byte-equality, field-set or uniqueness failure occurred.
- Repair actions: none required this round. Original corpus untouched; no previously committed batch was modified; no teacher-A path was read, opened or grepped.
- Final schema check: VERIFY_PASS batch=10 aggregate=2380 prefix=ok ids_unique=ok - JSONL line-parseable and newline-terminated, 10 rows, exactly the 12 required fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, source_user/source_assistant byte-equal to corpus by role, corrected_answer non-empty with unique stance headers within the batch, quality_dimensions three ints in [1,5] with bool excluded, risks/evidence_required non-empty string arrays, confidence float in [0,1], global source_id uniqueness, aggregate 2380 rows a strict prefix of train.jsonl, no validation-batch files.
- Manifest: MANIFEST.sha256 regenerated over all files in this experiment directory except MANIFEST.sha256 itself; sha256sum -c reported all OK.

Technical topics covered by this batch. All ten source items instantiate the same degenerate template - an agent that repeatedly calls a calculator when the answer is already known, scenario variants 112 through 123 (the corpus omits variants 113 and 115) - each paired with an assistant turn that is a grading rubric rather than an actual answer, so all ten are rewrites. The ten rewrites take ten distinct, non-overlapping stances that do not repeat those used in rounds 0234-0237: (v112) counterfactual rather than heuristic definition of an unnecessary call, via no-op replay under fixed seeds, with the boundary condition that replay-based labelling is valid only for side-effect-free tools; (v113 slot -> v114) separating decision error from execution distrust using exact-repeat argument-tuple tagging, and the stale-result hazard introduced by a dedup cache over stateful tools; (v116) cost accounting in added tokens, added seconds and GPU-seconds per solved task rather than raw call counts, with the warning that GPU-second deltas can be a batch-occupancy artifact; (v117) the confounder set - task-mix drift, tool-latency drift, template drift - handled with a 2x2 frozen/unfrozen design whose negative-control cell is a stop gate on the measurement rig itself; (v118) preference-signal design, arguing correctness-tied pairwise preference over a scalar correctness-minus-lambda-calls reward, with a KL-to-reference budget as an independent rollback trigger; (v119) the stop / no-tool evaluation as a standalone probe, scored jointly with a tool-required slice through a routing confusion matrix so refusal-everywhere cannot score well; (v120) gateway-level canary mechanics with sticky per-session arm assignment sharing model replicas and KV-cache pressure across arms, plus explicit automatic rollback triggers; (v121) statistical power, deriving an ESTIMATE of roughly 200 paired items for a 0.3-call detectable difference and establishing that the correctness endpoint can only support a non-inferiority claim, not equivalence; (v122) the failure taxonomy and conditional recovery rate under seeded fault injection, treating suppressed legitimate retries as the primary hazard of any call-reduction intervention; (v123) an end-to-end four-step decision protocol with per-step kill criteria and a sealed step-1 prediction diffed against the MEASURED canary delta.

Every rewrite states its assumptions explicitly, gives the mechanism, lists the boundary conditions that flip the recommendation, states a falsifiable hypothesis with its refutation condition, specifies a single-variable controlled experiment, enumerates the evidence artifacts required to adjudicate it, and defines a rollback or kill gate. No quantitative value in this batch is labelled MEASURED; every number is labelled ESTIMATE and carries its derivation, because no run was executed for this review.

Status caveat. These outputs are provisional teacher-B review material produced under blind review - no teacher-A artifact was read, opened or grepped at any point while producing this batch. They are not expert gold, they have not been adjudicated by a human expert, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step outside the scope of this worker.
"""

txt = open(MD).read()
assert txt.startswith(TITLE)
rest = txt[len(TITLE):]
open(MD, "w").write(TITLE + entry + rest)
print("MD_UPDATED")
