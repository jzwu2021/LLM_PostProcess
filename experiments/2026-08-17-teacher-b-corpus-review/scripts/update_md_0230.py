import re
P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
NEW = """## Run 0231 (train-batch-0230.jsonl)

- Batch file: results/train-batch-0230.jsonl
- Corpus range: train.jsonl positional slice [2290:2300]
- Source IDs: corpus-02521, corpus-02523, corpus-02524, corpus-02525, corpus-02527, corpus-02528, corpus-02529, corpus-02530, corpus-02531, corpus-02532 (10 items in corpus order; taken by positional slice, not by ID arithmetic - the IDs in this window are non-consecutive - and corpus order is preserved exactly).
- Progress: 2300/2500 train (92.0%); remaining 200. Validation target is 0 by user instruction; no validation-batch file exists or was created.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run of scripts/tb_verify_batch_0230.py (JSONL line-parse, count=10, all 12 required fields, teacher_lane/teacher_model/calibration_status/decision values, byte-exact source_user and source_assistant against corpus, non-empty corrected_answer distinct from source_assistant, ESTIMATE label present, stance marker present, quality_dimensions integer 1-5, non-empty risks and evidence_required, confidence in [0,1], global source_id uniqueness, aggregated sequence is a strict prefix of train.jsonl, zero validation-batch files).
- Repairs: none to the batch output. One process-level fix was made before generation: the batch generator was restructured so the ten stance bodies live in a separate module (scripts/stances_0230.py) imported by scripts/gen_batch_0230.py, rather than being inlined as in earlier batches; a stray placeholder import left by that restructuring was removed before the generator was first executed. The generator then ran once, and the verifier - derived from the previous batch's verifier by sed substitution of the batch number - passed on its first execution. No batch file was rewritten, no prior batch was touched, and neither the original corpus nor any teacher-A artifact was read or modified.
- Final schema check: VERIFY_PASS, TOTAL 2300.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except MANIFEST.sha256 itself; `sha256sum -c` reports all entries OK.

Technical topics covered by this batch. All ten items are further variants of the same prompt family - an agent that repeatedly calls a calculator when the answer is already known, requiring metrics plus an intervention, with an explicit falsifiable hypothesis and a controlled experiment. Each corpus item's assistant turn is a grading rubric rather than an answer, so every item is a rewrite. The shared frame is unchanged: three distinct causes of redundant tool calls (instruction-driven, policy-driven, attention-driven), a minimum metric set (unnecessary-call rate, tool success, final correctness, trajectory length, tool latency, recovery), and an intervention ladder ordered by reversibility. The ten rewrites are differentiated by ten new analytical stances:

- Stance 300: before optimizing the behaviour, bound the achievable saving with a tool-deregistration ablation reported per pre-registered arithmetic difficulty bucket; a uniform pooled correctness figure can hide collapse confined to the multi-step bucket.
- Stance 301: redundancy must be defined against a counterfactual the model could actually execute, so labels require a direct-sampling capability probe at the pinned decoding configuration; capability-anchored labels are checkpoint-specific and expire when the checkpoint changes.
- Stance 302: abstention must be a first-class action with its own probability, log line and paired evaluation sets (abstain-correct and must-call), because prompt and reward interventions shift a threshold rather than improving discrimination.
- Stance 303: caching reframes the problem from policy to serving, removing the tool round-trip cost while leaving the extra model turn - prompt reprocessing, output tokens and KV cache occupancy - fully intact, so it is a partial fix that must be measured with a decomposed latency report.
- Stance 304: prompt-level fixes are fragile because the system prompt is a shared mutable artifact; durability requires prompt hashing plus blocking pre-merge checks whose thresholds are set against the measured noise band of the evaluation sets.
- Stance 305: preference training is the last rung of the ladder because it is the only one that cannot be reverted by a config change; preference pairs must hold the final answer fixed so the signal cannot be satisfied by shortening the answer.
- Stance 306: recovery is the property most likely to be destroyed and least likely to be measured, since self-correction depends on exactly the extra call being suppressed; it needs an injected-error adversarial set with a pre-registered floor, because natural traffic contains too few recovery episodes.
- Stance 307: segment before aggregating - a pooled redundancy rate is a mix-weighted mean, so traffic-mix shift alone produces apparent effects, and promotion must require improvement within each pre-registered segment.
- Stance 308: a result is a function of seven artifacts (trace, prompt, schema, checkpoint, decoding configuration, evaluation sets, label set); runs missing any identifier are excluded from promotion decisions rather than interpreted with caveats.
- Stance 309: the honest closing position is that the source pair supplies no answer at all; the low instruction-coverage score is a property of the rubric-shaped response and not of the prompt, which is the scarce and worth-keeping half of the item.

Every quantitative claim in this batch is labelled ESTIMATE with its derivation. No value is labelled MEASURED, because no benchmark was executed for this review. These outputs are provisional teacher-B review material produced blind to the teacher-A lane. They are not expert gold, they have not been adjudicated against teacher-A, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step and was not performed here.

"""
s = open(P).read()
marker = "## Run 0230 (train-batch-0229.jsonl)"
assert marker in s
i = s.index(marker)
s = s[:i] + NEW + s[i:]
open(P, "w").write(s)
print("OK")
