import io, os
EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
MD = f"{EXP}/EXPERIMENT.md"

entry = """## Round 0234 - train-batch-0234.jsonl

- Batch file: results/train-batch-0234.jsonl
- Corpus interval (positional): train.jsonl rows [2330, 2340)
- Source IDs: corpus-02571 .. corpus-02580 (10 items, contiguous, corpus order preserved)
- Progress: train 2340/2500 (remaining 160); validation 0/0 by user instruction, no validation batch files exist or were created
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_0234.py) - JSONL line-parseable, 10 rows, 12 required fields exactly, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, source_user/source_assistant byte-equal to corpus, corrected_answer non-empty, quality_dimensions three ints in [1,5] with bool excluded, risks/evidence_required non-empty string arrays, confidence float in [0,1], global source_id uniqueness, aggregate 2340 rows a strict prefix of train.jsonl, no validation-batch files.
- Repair actions: none required this round.
- Final schema check: VERIFY_PASS batch=10 aggregate=2340 prefix=ok ids_unique=ok
- Manifest: MANIFEST.sha256 regenerated over all files in this experiment directory except MANIFEST.sha256 itself; sha256sum -c reported all OK.

Technical topics covered by this batch. All ten source items are the same degenerate template - an agent-reliability prompt about an agent that repeatedly calls a calculator when the answer is already known, scenario variants 71 through 80, each paired with an assistant turn that is a grading rubric rather than an answer. All ten are therefore rewrites. The ten rewrites take ten distinct, non-overlapping analytical stances: (340) the accuracy-cost Pareto frontier and the degenerate zero-call optimum of any single-scalar frugality objective; (341) redundancy as a mixture concentrated in a minority of prompt templates, making decomposition the first diagnostic and a targeted template edit the cost-effective fix; (342) the necessity-stratified stop-or-no-tool held-out set as the only instrument powered to separate over-suppression from aggregate accuracy noise; (343) preference and reward optimization as the least reversible intervention, its global effect on tool-call behavior outside the preference distribution, and checkpoint-level rollback mechanics; (344) trajectory length as a confounder rather than a metric, normalization per call opportunity, stratification and reweighting; (345) tool success rate and tail latency instrumentation to separate retry-after-failure from genuine policy redundancy; (346) serving-configuration pinning - decoding temperature, model version, schema and prompt hashes - plus multi-seed variance as a precondition for believing any small effect; (347) offline replay as evidence about mechanism but a biased forecast of production impact, with the live traffic split as the true promotion gate; (348) per-call provenance logging as the prerequisite for an auditable redundancy definition, with its latency, log-volume and redaction budgets; (349) the honest default that redundancy is a cost problem and not a correctness problem, which places the burden of proof on the intervention and makes a recorded no-action decision a legitimate result.

Every rewrite states its assumptions explicitly, gives the token-level mechanism by which tool-call emission arises, lists the boundary conditions that flip the recommendation, states one numbered falsifiable hypothesis with its refutation condition, specifies a single-variable controlled experiment, enumerates the evidence artifacts required to adjudicate it, and defines a rollback gate. No quantitative value in this batch is labelled MEASURED; every estimate is labelled ESTIMATE and carries its derivation, because no run was executed for this review.

Status caveat. These outputs are provisional teacher-B review material produced under blind review - no teacher-A artifact was read, opened or grepped at any point while producing this batch. They are not expert gold, they have not been adjudicated by a human expert, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step outside the scope of this worker.

"""

md = open(MD).read()
lines = md.split("\n")
# insert after the leading title block: find first line starting with '## '
idx = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
new = "\n".join(lines[:idx]) + "\n" + entry + "\n".join(lines[idx:])
open(MD, "w").write(new)
print("INSERTED at line", idx)
