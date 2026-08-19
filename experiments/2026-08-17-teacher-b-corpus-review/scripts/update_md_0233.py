P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
NEW = """## Run 0234 (train-batch-0233.jsonl)

- Batch file: results/train-batch-0233.jsonl
- Corpus range: train.jsonl positional slice [2320:2330]
- Source IDs: corpus-02558, corpus-02559, corpus-02561, corpus-02562, corpus-02563, corpus-02565, corpus-02566, corpus-02567, corpus-02568, corpus-02570 (10 items in corpus order; taken by positional slice, not by ID arithmetic - the ID sequence is non-consecutive and corpus order is preserved exactly).
- Progress: 2330/2500 train (93.2%); remaining 170. Validation target is 0 by user instruction; no validation-batch file exists or was created.
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run of scripts/verify_0233.py (JSONL line-parse and trailing newline, count=10, exact 12-field set with no extras, teacher_lane/teacher_model/calibration_status/decision values, positional source_id match, byte-exact source_user and source_assistant against corpus, non-empty corrected_answer, quality_dimensions integer 1-5 with bool excluded, non-empty risks and evidence_required, confidence float in [0,1] with bool excluded, stance-header uniqueness within batch, global source_id uniqueness, aggregated sequence is a strict prefix of train.jsonl, aggregate count 2330, zero validation-batch files).
- Repairs: none. The stance data was authored fresh as scripts/tb_stances_0233.py with ten new stances (330-339) and imported by scripts/tb_gen_batch_0233.py; the verifier scripts/verify_0233.py was derived from the previous run's verifier by sed substitution, and all three hardcoded offsets (batch filename, positional slice bounds, aggregate count) were corrected in one pass before execution. Generator ran once, verifier passed on first execution. No batch file was rewritten, no prior batch was touched, and neither the original corpus nor any teacher-A artifact was read or modified.
- Final schema check: VERIFY_PASS, aggregate=2330, prefix=ok, ids_unique=ok.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except MANIFEST.sha256 itself; `sha256sum -c` reports all entries OK.

Technical topics covered by this batch. All ten items belong to the agent-reliability prompt family - an agent that repeatedly calls a calculator when the answer is already known, requiring metrics plus an intervention, with an explicit falsifiable hypothesis and a controlled experiment. Each corpus item's assistant turn is a grading rubric rather than an answer, so every item is a rewrite. The shared frame states the tool-invocation mechanism as a token-level policy decision, the superlinear cost of an appended tool result within a trajectory, and three boundary conditions that flip the recommendation. The ten stances differentiate along measurement validity, intervention reversibility and decision economics:

- Stance 330: redundancy must be defined by an adjudicated labelling rule before any metric exists, because an argument-equality proxy counts legitimate re-verification as waste and can invert an intervention's measured sign.
- Stance 331: cost is superlinear within a trajectory, since an appended tool result is re-read as prefill on every later turn; position-weighted accounting is required and call-count reduction is not a cost proxy.
- Stance 332: a memoization cache is the cheapest and most reversible intervention but converts latency into staleness, so tool purity must be audited rather than assumed.
- Stance 333: prompt and tool-description edits are cheap but checkpoint-specific, so the prompt hash must be pinned to the model checkpoint hash it was validated against.
- Stance 334: fine-tuning against a call-count objective teaches indiscriminate suppression and regresses exactly the arithmetic tasks the tool was added to serve; it must be gated on held-out necessary-call recall.
- Stance 335: redundancy is concentrated rather than uniform, so a headline rate hides the segment where a targeted template fix would capture most of the achievable gain.
- Stance 336: tool latency and failure behavior are confounders - the same policy is nearly free behind a local calculator and catastrophic behind a slow, rate-limited remote tool where retries compound.
- Stance 337: sampled decoding makes per-trajectory redundancy a random variable, so paired designs with a justified repeat count are required and effects below the measured spread are null results.
- Stance 338: continuous segmented monitoring with covariates is part of the deliverable, since the intervention's binding conditions erode as prompts and checkpoints change on independent cadences.
- Stance 339: the decision belongs on cost per successful task rather than redundancy rate, because an intermediate metric invites suppression whose accuracy cost lands on someone else's dashboard.

Every quantitative claim in this batch is labelled ESTIMATE with its derivation. No value is labelled MEASURED, because no benchmark was executed for this review. These outputs are provisional teacher-B review material produced blind to the teacher-A lane. They are not expert gold, they have not been adjudicated against teacher-A, and they are not evidence about any model's domain capability. Agreement analysis against teacher-A is a separate, later step and was not performed here.

"""
s = open(P).read()
marker = "## Run 0233 (train-batch-0232.jsonl)"
assert marker in s, "marker not found"
i = s.index(marker)
s = s[:i] + NEW + s[i:]
open(P, "w").write(s)
print("OK")
