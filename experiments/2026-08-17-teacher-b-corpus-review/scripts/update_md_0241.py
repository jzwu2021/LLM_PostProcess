import io

P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
HDR = "# Experiment: teacher-B corpus review (blind, independent second opinion)\n"

NEW = """
## Round 0241 - train-batch-0241.jsonl

- Batch file: results/train-batch-0241.jsonl
- Corpus interval (positional): train.jsonl rows [2400, 2410)
- Source IDs: corpus-02649 .. corpus-02658 (10 items, corpus order preserved; this slice is ID-consecutive)
- Progress: train 2410/2500 (remaining 90); validation 0/0 by user instruction, no validation batch files exist or were created
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_0241.py, derived by sed from scripts/verify_0239.py with three hardcoded offsets changed together - the batch filename 0239 -> 0241, the positional window [2380,2390) -> [2400,2410), and the aggregate-count assertion 2390 -> 2410). The generator selected message content by role, so no byte-equality, field-set, ordering or uniqueness failure occurred.
- Repair actions: none required this round. The original corpus was not modified; no previously committed batch was edited; no teacher-A path under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, grepped or listed at any point.
- Final schema check: VERIFY_PASS batch=10 aggregate=2410 prefix=ok ids_unique=ok - JSONL newline-terminated and line-parseable, exactly 10 physical lines, all 12 required fields present and no extras, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, source_user/source_assistant byte-equal to the corpus values selected by role, corrected_answer non-empty with unique stance headers within the batch, quality_dimensions three ints in [1,5] with bool excluded, risks/evidence_required non-empty string arrays, confidence float in [0,1], global source_id uniqueness across the aggregate, aggregate 2410 rows a strict prefix of train.jsonl, and no validation-batch files present.
- Manifest: MANIFEST.sha256 regenerated over every file in this experiment directory except MANIFEST.sha256 itself; sha256sum -c reported all OK.

Technical topics covered by this batch. All ten source items are the same degenerate template - an agent that repeatedly calls a calculator when the answer is already known, scenario variants 149 through 158 - paired with an assistant turn that is a grading rubric rather than an answer, so all ten are rewrites. The ten rewrites take ten distinct and mutually non-overlapping engineering stances: (1) redundancy as a decision-boundary defect in the tool-vs-answer action prior, with a three-arm prompt-only vs SFT-remix design and an explicit sample-size derivation labelled ESTIMATE; (2) cost control via a trajectory-scoped memoisation guard at the router, gated on duplicate-expression share measured from router logs and on trajectory-scoped rather than global cache keys to avoid cross-tenant leakage; (3) measurement validity, arguing that UCR is unfalsifiable until the "already known" label is fixed by convention, requiring both the verbatim-repeat proxy and the no-tool reference-model proxy plus Cohen's kappa; (4) harness blindness, adding no-tool and stop-behaviour eval subsets before any training change and gating on cross-run reproducibility within 2 absolute points; (5) preference optimisation with an explicit reward-hacking failure mode, where the objective can be satisfied by silence, requiring length-controlled pairs, a KL budget and an under-calling guardrail; (6) the serving-side capacity view, insisting that continuous-batching cost is load-dependent so capacity claims must come from a load sweep to SLO breach rather than single-stream latency; (7) the contrarian position that confirming and correcting redundancy are different behaviours and that blanket suppression can delete a real verification safety net, falsifiable offline from logged trajectories before any model change; (8) operational safety of rollout, where tool-call presence is an audit artifact for some tenants so a silent global default is a compliance regression that fleet averages would hide; (9) data root-cause, where the tic is inherited from a teacher prompted to always show work with tools, killable cheaply by a corpus density audit before any retrain; and (10) synthesis with an ordering rule by reversibility rather than expected effect size, with per-stage rollback gates.

Every rewrite names the mechanism, states a numbered falsifiable hypothesis with an explicit falsification condition, specifies a single-variable controlled experiment, enumerates the confounders, lists the evidence artifacts required to adjudicate it, and defines a rollback gate. Every numeric figure is explicitly labelled ESTIMATE or MEASURED with its derivation shown - the sample-size figures are ESTIMATE from the normal approximation and are stated as such, and no MEASURED value is claimed anywhere in this batch because no run was executed for this review.

Status caveat. These outputs are provisional teacher-B review material produced under blind review. They are not expert gold, they have not been adjudicated by a human expert, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step outside the scope of this worker.
"""

s = open(P).read()
assert s.startswith(HDR), "header mismatch"
rest = s[len(HDR):]
open(P, "w").write(HDR + NEW + rest)
print("MD_UPDATED")
