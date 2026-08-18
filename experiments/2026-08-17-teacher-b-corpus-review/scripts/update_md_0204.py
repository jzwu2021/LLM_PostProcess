import io, re

EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
MD = f"{EXP}/EXPERIMENT.md"

ENTRY = """## Run 0204 - train-batch-0204.jsonl (provisional teacher-B blind review)

- Batch file: results/train-batch-0204.jsonl
- Corpus range: research/ai-infra-expert/corpus/train.jsonl positional lines 2031-2040 (0-indexed slice 2030:2040)
- Source IDs: corpus-02240 .. corpus-02250 in corpus order (note: corpus-02246 is absent from train.jsonl, so the ten items are the ten consecutive lines, not a contiguous numeric range; positional slicing was used, never ID arithmetic)
- Progress: train 2040/2500 (stage target 2500, set by the user on 2026-08-18, replacing the original 6000 full-corpus target). Validation target is 0; no validation-batch file exists or was created.
- Remaining to stage target: 460
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: PASS on first run (scripts/tb_verify_batch_0204.py, derived from the 0203 verifier by sed with only the batch number and slice offset changed). No repair actions were needed this round.
- Repairs: none.
- Final schema check: VERIFY_PASS, aggregate TOTAL 2040, aggregated source_id sequence confirmed to be a strict prefix of train.jsonl with no duplicates; all 12 required fields present; teacher_lane/teacher_model/calibration_status/decision values correct; source_user and source_assistant byte-identical to corpus; every corrected_answer non-empty, distinct within the batch, and distinct from the source assistant text; confidence in [0,1]; zero validation-batch files present.
- Ad-hoc cross-batch control: the ten stance headline sentences in this batch were compared against every stance headline in all previously written batches; 10 new, 0 overlap, confirming no stance reuse across the homogeneous TP-versus-PP block.
- Manifest: MANIFEST.sha256 regenerated over all files in this experiment directory except MANIFEST.sha256 itself and __pycache__; sha256sum -c verified with zero failures.

Technical topics covered by this batch: this is a further homogeneous block of the tensor-parallel versus
pipeline-parallel family (scenario variants 40-50), whose corpus assistant turns contain only a grading rubric rather
than an answer. The batch was therefore written as ten mutually disjoint analytical stances over one shared assumption
and mechanism frame, so that ten near-identical prompts do not yield ten near-identical answers. The stances are:
attention-head and hidden-dimension divisibility as the constraint that decides which TP degrees are even legal, plus
KV-head replication under grouped-query attention turning an expected memory win into a regression; fleet-level
capacity and availability planning where GPUs spent on shrinking one replica are GPUs not spent on another replica,
evaluated under N-1; kernel and library maturity, where silent attention-backend fallbacks and non-overlapped
collectives can exceed the topology effect being measured; long-context regimes where per-step KV traffic, not weight
traffic, sets the bottleneck and TP keeps dividing it while PP does not; energy and thermal steady state, where boost
clocks in a short run misrepresent sustained serving behaviour; request heterogeneity and per-cohort tail fairness
under a replayed production trace rather than a uniform synthetic load; startup, weight-load, warmup and
crash-recovery cost weighted by deploy and autoscale churn rather than request rate; observability adequacy as a
precondition, requiring a per-step time-attribution budget that reconciles to end-to-end step time before any
topology claim is falsifiable; portability and lock-in, where checkpoint shard layout encodes a topology assumption
and the crossover point does not transfer across fabric or model shape; and statistical adequacy, requiring an A/A
noise-floor run and a pre-registered equivalence margin before any reported TP-versus-PP gap may be interpreted.

Every quantitative claim in this batch is explicitly labelled ESTIMATE and carries its derivation. No value is labelled
MEASURED, because no benchmark was executed for this review. These outputs are provisional teacher-B review material
produced blind (teacher-A artifacts were not read at any point during generation). They are not expert gold, they have
not been validated by a human domain expert, and they are not evidence about any model's domain capability.

"""

src = open(MD).read()
marker = "## Run 0203 - "
i = src.index(marker)
open(MD, "w").write(src[:i] + ENTRY + src[i:])
print("EXPERIMENT.md updated")
