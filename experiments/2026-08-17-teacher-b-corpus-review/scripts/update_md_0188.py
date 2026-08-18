import re
P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s = open(P).read()
entry = """## Run 2026-08-18 batch 0188

- Batch file: results/train-batch-0188.jsonl
- Corpus range: train.jsonl positional lines 1871-1880, ten physically consecutive rows, original
  corpus order preserved, nothing skipped or reordered. Selection by line position, not ID arithmetic
  (the window contains ID gaps: corpus-02069 and corpus-02071 are absent from the corpus).
- Source IDs: corpus-02064, 02065, 02066, 02067, 02068, 02070, 02072, 02073, 02074, 02075.
- Progress: 1880/2500 train records (75.2%). Remaining: 620. The 2500 denominator is the user-set
  staged target adopted on 2026-08-18, replacing the original 6000-record figure. Validation target
  is 0 for this stage; zero validation-batch files exist and none were created (verifier asserts it).
- Decisions: keep 0, rewrite 10, reject 0.
- Initial schema check: PASS on the first run of scripts/tb_verify_batch_0188.py, derived from the
  batch-0187 verifier by literal 0187->0188 substitution rather than rewritten. Checks: per-line JSON
  parse, batch count 10, all 12 required fields, fixed values for teacher_lane / teacher_model /
  calibration_status, decision in {keep,rewrite,reject}, byte-exact equality of source_user and
  source_assistant against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer,
  confidence in [0,1], global source_id uniqueness across all batches, and the aggregate train
  sequence being a strict prefix of train.jsonl (1880 records).
- Repair actions: none required this run.
- Final schema check: PASS (VERIFY_PASS).
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself;
  sha256sum -c reports all files OK.
- Technical topics covered: all ten prompts are variants of the same weight-only quantization
  fair-comparison scenario (variants 164-175), so the ten answers are deliberately given ten distinct
  analytical stances rather than ten paraphrases: memory-bandwidth roofline prediction and
  measured-versus-predicted reconciliation; KV-cache accounting and whether freed HBM actually became
  concurrency; reproducibility and full provenance pinning including driver, image and kernel-library
  hashes; workload representativeness via replayed production arrival traces versus fixed-length
  synthetic microbenchmarks; statistical power, pre-registered sample size and A/A noise floor;
  quality-metric validity established by positive control against a knowingly degraded checkpoint;
  rollout mechanics with concurrent dual-path canaries and pre-registered abort thresholds;
  generalisation boundaries across model family, bit width, group size, engine and accelerator
  generation; failure-mode enumeration verified by fault injection rather than inspection; and the
  auditable decision record as the actual deliverable. Every numeric claim in the batch is labelled
  ESTIMATE with an inline derivation or MEASURED with an artifact reference.
- Status: PROVISIONAL. These are teacher-B second-opinion rewrites produced blind, without any access
  to the teacher-A calibration directory. They are NOT expert gold, they have not been adjudicated
  against teacher-A, and they say nothing about any model's domain capability. Agreement analysis is a
  separate later step outside this worker's scope.

"""
s = s.replace("## Run 2026-08-18 batch 0187", entry + "## Run 2026-08-18 batch 0187", 1)
open(P, "w").write(s)
print("OK")
