EXP = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
MD = f"{EXP}/EXPERIMENT.md"

ENTRY = """## Run 0205 - train-batch-0205.jsonl (provisional teacher-B blind review)

- Batch file: results/train-batch-0205.jsonl
- Corpus range: research/ai-infra-expert/corpus/train.jsonl positional lines 2041-2050 (0-indexed slice 2040:2050)
- Source IDs: corpus-02251 .. corpus-02260, ten consecutive corpus lines taken by positional slicing, never ID arithmetic
- Progress: train 2050/2500 (stage target 2500, set by the user on 2026-08-18, replacing the original 6000 full-corpus target). Validation target is 0; no validation-batch file exists or was created.
- Remaining to stage target: 450
- Decisions: keep 0 / rewrite 10 / reject 0
- Initial schema check: PASS on first run (scripts/tb_verify_batch_0205.py, derived from the 0204 verifier by sed with only the batch number changed). No repair actions were needed this round.
- Repairs: none.
- Final schema check: VERIFY_PASS, aggregate TOTAL 2050, aggregated source_id sequence confirmed to be a strict prefix of train.jsonl with no duplicates; all 12 required fields present; teacher_lane/teacher_model/calibration_status/decision values correct; source_user and source_assistant byte-identical to corpus; every corrected_answer non-empty, distinct within the batch, and distinct from the source assistant text; confidence in [0,1]; zero validation-batch files present.
- Ad-hoc cross-batch control: the ten stance headline sentences in this batch were checked against the stance headlines already used in the preceding TP-versus-PP block; all ten are new, continuing the disjoint-stance strategy for this homogeneous family.
- Manifest: MANIFEST.sha256 regenerated over all files in this experiment directory except MANIFEST.sha256 itself and __pycache__; sha256sum -c verified with zero failures.

Technical topics covered by this batch: a further homogeneous block of the tensor-parallel versus pipeline-parallel
family (scenario variants 51-60), whose corpus assistant turns again contain only a grading rubric rather than an
answer. Ten mutually disjoint analytical stances were written over one shared assumption and mechanism frame: NVLink
link-state, driver and firmware health as a silent precondition where a degraded fabric biases the comparison toward
PP through unannounced NCCL transport fallback; admission control and preemption policy, where one arm crossing the
KV-exhaustion threshold turns a preemption artifact into an apparent parallelism effect; the security and isolation
surface, where cross-node PP moves hidden-state activations off-host and changes the data-protection posture without
changing model output; partial-failure recovery, where mean time to restore capacity, dominated by per-process weight
load, matters more than mean token latency; replica-set-level comparison, where the deployable question is which
factorisation of a fixed device budget meets the SLO with the fewest devices once queueing delay dominates the tail;
harness self-reproducibility, requiring the repeatability band of repeated identical runs to be published before any
inter-layout difference is interpreted; allocator fragmentation and reserve behaviour, where arithmetic KV capacity is
an upper bound and the shortfall differs by layout; decision framing, where the tolerance and reversibility of the
actual decision determine the required evidence bar and may make the layouts decision-equivalent; the cost of the
comparison itself, with a pre-registered reduced design and an explicit planned stop; and an explicit statement of the
review's own limits, including a blind second-lane agreement design that must not be fed back into generation.

Every quantitative claim in this batch is explicitly labelled ESTIMATE and carries its derivation. No value is labelled
MEASURED, because no benchmark was executed for this review. These outputs are provisional teacher-B review material
produced blind (teacher-A artifacts were not read at any point during generation). They are not expert gold, they have
not been validated by a human domain expert, and they are not evidence about any model's domain capability.

"""

src = open(MD).read()
marker = "## Run 0204 - "
i = src.index(marker)
open(MD, "w").write(src[:i] + ENTRY + src[i:])
print("EXPERIMENT.md updated")
