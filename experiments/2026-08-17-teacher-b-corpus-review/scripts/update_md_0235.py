EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
MD = f"{EXP}/EXPERIMENT.md"

entry = """## Round 0235 - train-batch-0235.jsonl

- Batch file: results/train-batch-0235.jsonl
- Corpus interval (positional): train.jsonl rows [2340, 2350)
- Source IDs: corpus-02581 .. corpus-02590 (10 items, contiguous, corpus order preserved)
- Progress: train 2350/2500 (remaining 150); validation 0/0 by user instruction, no validation batch files exist or were created
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: FAIL on first run (scripts/verify_0235.py) - all 20 byte-equality assertions failed (source_user and source_assistant for rows 0..9). Root cause: the generator read messages by positional index (messages[0]=user, messages[1]=assistant), but records in this corpus segment carry three messages (system, user, assistant), so the generator captured the system prompt as source_user and the user turn as source_assistant.
- Repair actions: rewrote the generator to select message content by role (dict keyed on x["role"]) instead of by list position, and corrected the review note in each corrected_answer, since the actual assistant turn is a grading rubric rather than the task statement the positional bug had shown. Original corpus untouched; no previously committed batch was modified. Batch 0235 regenerated and re-verified.
- Final schema check: VERIFY_PASS batch=10 aggregate=2350 prefix=ok ids_unique=ok - JSONL line-parseable and newline-terminated, 10 rows, exactly the 12 required fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, source_user/source_assistant byte-equal to corpus by role, corrected_answer non-empty, quality_dimensions three ints in [1,5] with bool excluded, risks/evidence_required non-empty string arrays, confidence float in [0,1], global source_id uniqueness, aggregate 2350 rows a strict prefix of train.jsonl, no validation-batch files.
- Manifest: MANIFEST.sha256 regenerated over all files in this experiment directory except MANIFEST.sha256 itself; sha256sum -c reported all OK.

Technical topics covered by this batch. All ten source items are the same degenerate template - an agent-reliability prompt about an agent that repeatedly calls a calculator when the answer is already known, scenario variants 81 through 90 - each paired with an assistant turn that is a grading rubric ("Answer should state assumptions, a falsifiable hypothesis, measurements, expected confounders, and rollback criteria...") rather than an answer. All ten are therefore rewrites. The ten rewrites take ten distinct, non-overlapping analytical stances: (2340) a precise operational definition of redundancy via canonicalized argument hashing plus an in-window visibility flag, and the detection metrics that follow from it; (2341) separating context-eviction-driven redundancy from policy-habit redundancy by holding the model fixed and varying only the context budget; (2342) a per-trajectory memoization interceptor scoped strictly to tools declared PURE as the lowest-risk intervention, with task success as the guard metric; (2343) cost and latency accounting for redundant calls in wasted decode turns and GPU seconds, with every figure explicitly labelled ESTIMATE and its derivation shown; (2344) the prompt-only intervention, its cheapness, and its structural ceiling at the visible-redundancy subset; (2345) false suppression as the dangerous failure mode, and shadow-execute-and-compare as the only way to validate a purity declaration; (2346) the training-side intervention via paired preference data with loss masked to assistant/tool-call segments, gated on a repeated-baseline noise band for unrelated tool use; (2347) the trace schema that must be fixed in advance for any of these metrics to be reproducible and auditable offline; (2348) staged rollout with pre-registered rollback gates at each traffic step; (2349) threats to validity - concurrent arms, trajectory-level randomization, pre-registered minimum detectable effect, and the rule that a single run without a confidence interval is not evidence.

Every rewrite states its assumptions explicitly, gives the mechanism, lists boundary conditions, states a falsifiable hypothesis with its refutation condition, specifies a single-variable controlled experiment, enumerates the evidence artifacts required, and defines a rollback gate. No quantitative value in this batch is labelled MEASURED; the one arithmetic figure is labelled ESTIMATE and carries its derivation, because no run was executed for this review.

Status caveat. These outputs are provisional teacher-B review material produced under blind review - no teacher-A artifact was read, opened or grepped at any point while producing this batch. They are not expert gold, they have not been adjudicated by a human expert, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step outside the scope of this worker.

"""

md = open(MD).read()
lines = md.split("\n")
idx = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
new = "\n".join(lines[:idx]) + "\n" + entry + "\n".join(lines[idx:])
open(MD, "w").write(new)
print("INSERTED at line", idx)
