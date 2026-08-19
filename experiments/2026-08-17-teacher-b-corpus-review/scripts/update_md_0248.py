P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
HDR = "# Experiment: teacher-B corpus review (blind, independent second opinion)\n"
txt = open(P).read()
assert txt.startswith(HDR)
rest = txt[len(HDR):]

entry = """## Round 0248 - train-batch-0248.jsonl

- Batch file: results/train-batch-0248.jsonl
- Corpus interval (positional): train.jsonl rows [2470, 2480)
- Source IDs: corpus-02724 .. corpus-02734 (10 items, corpus order preserved; the slice was taken by positional slicing of train.jsonl, never by ID arithmetic, so the gap at corpus-02732 is absorbed transparently)
- Progress: train 2480/2500 (remaining 20); validation 0/0 by user instruction, no validation batch files exist or were created
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (verify_0248.py, derived by sed from scripts/verify_0246.py with the three hardcoded offsets changed together - batch filename 0246 -> 0248, positional window [2450,2460) -> [2470,2480), and the aggregate-count assertion 2460 -> 2480). The generator selected message content by role, so no byte-equality, field-set, ordering or uniqueness failure occurred.
- Repair actions: none. No fix or regeneration was required this round. The original corpus was not modified; no previously committed batch was edited; no teacher-A path under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, grepped or listed at any point in this round.
- Final schema check: VERIFY_PASS batch=10 aggregate=2480 prefix=ok ids_unique=ok - JSONL newline-terminated and line-parseable, exactly 10 physical lines, all 12 required fields present and no extras, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, source_user/source_assistant byte-equal to the corpus values selected by role, corrected_answer non-empty with unique stance headers within the batch, quality_dimensions three ints in [1,5] with bool excluded, risks/evidence_required non-empty string arrays, confidence float in [0,1], global source_id uniqueness across the aggregate, aggregate 2480 rows a strict prefix of train.jsonl, and no validation-batch files present.
- Manifest: MANIFEST.sha256 regenerated over every file in this experiment directory except MANIFEST.sha256 itself (__pycache__ excluded); sha256sum -c reported all OK.

Technical topics covered by this batch. All ten source items continue the degenerate calculator template - an agent that repeatedly calls a calculator when the answer is already known, scenario variants 224 through 234 - paired with an assistant turn that is a grading rubric rather than an answer, so all ten are rewrites. The ten rewrites take ten distinct stances (51 through 60) that reuse no stance header from any previously committed round: (51) a stale-cache/representation framing where untyped tool observations carry no validity scope, tested by a typed-observation serializer swap with a length-matched control and window-occupancy logging; (52) a contrarian counterfactual-measurement argument that argument-matching produces a label with an unknown false-positive rate, tested by a deterministic fork/replay harness gated on a null-fork byte-identity proof; (53) admission control on a shared serving cluster, where redundant calls are prefill queueing load and a per-task quota is evaluated under randomized interleaved blocks with open/close baselines; (54) a linear-probe representation test at the decision position with shuffled-label control, paraphrase robustness and cross-schema generalization, used to decide whether weight-level work is even addressable; (55) deterministic content-addressed memoization outside the model as the cheapest reversible fix, scoped by an explicit tool-purity allowlist with a live-vs-cached agreement audit; (56) an anti-undercalling invariant on any reward or preference signal, with disjoint easy/hard slices reported separately and a decoded loss-mask audit as a precondition; (57) reversibility-ordered staging R1-R5 with pre-registered gates and the rule that no rung may begin before its predecessor's gate passes; (58) a metric-denominator objection replacing per-task with per-decision rates, resolved by cheap offline re-analysis of stored trajectories; (59) converting the finding into a calibrated permanent regression gate whose false-failure rate is measured over ten unchanged runs before enforcement; (60) synthesis naming five load-bearing assumptions with the specific observation that would kill each, plus a program-level rollback if opening and closing baselines disagree.

Every rewrite names the mechanism, states a numbered falsifiable hypothesis (H1-H10), specifies a single-variable controlled experiment, enumerates confounders and boundary conditions, lists the evidence artifacts required to adjudicate it, and defines a rollback gate. Every numeric figure is explicitly labelled ESTIMATE with its derivation shown; no MEASURED value is claimed anywhere in this batch because no run was executed for this review, and stance 60 restates explicitly that the 20% redundancy rate, the 1.5k-token prompt, the ~310 tokens/task overhead and the ~4.5M-token gate cost are all ESTIMATEs that must be replaced by logged measurements before any external cost or capacity claim.

Status caveat. These outputs are provisional teacher-B review material produced under blind review. They are not expert gold, they have not been adjudicated by a human expert, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step outside the scope of this worker.



"""
open(P, "w").write(HDR + entry + rest)
print("MD_UPDATED")
