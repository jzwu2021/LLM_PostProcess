P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s = open(P).read()
entry = """## Run 2026-08-18 batch 0189

- Batch file: results/train-batch-0189.jsonl
- Corpus range: train.jsonl positional lines 1881-1890, ten physically consecutive rows, original
  corpus order preserved, nothing skipped or reordered. Selection by line position, not ID arithmetic
  (the window contains ID gaps: corpus-02083 and corpus-02087 are absent from the corpus).
- Source IDs: corpus-02077, 02078, 02079, 02080, 02081, 02082, 02084, 02085, 02086, 02088.
- Progress: 1890/2500 train records (75.6%). Remaining: 610. The 2500 denominator is the user-set
  staged target adopted on 2026-08-18, replacing the original 6000-record figure. Validation target
  is 0 for this stage; zero validation-batch files exist and none were created (verifier asserts it).
- Decisions: keep 0, rewrite 10, reject 0.
- Initial schema check: PASS on the first run of scripts/tb_verify_batch_0189.py, derived from the
  batch-0188 verifier by literal 0188->0189 substitution rather than rewritten. Checks: per-line JSON
  parse, batch count 10, all 12 required fields, fixed values for teacher_lane / teacher_model /
  calibration_status, decision in {keep,rewrite,reject}, byte-exact equality of source_user and
  source_assistant against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer,
  confidence in [0,1], distinct answer openings, global source_id uniqueness across all batches, and
  the aggregate train sequence being a strict prefix of train.jsonl (1890 records).
- Repair actions: none required this run.
- Final schema check: PASS (VERIFY_PASS).
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself;
  sha256sum -c reports all files OK.
- Technical topics covered: all ten prompts are variants of the same weight-only quantization
  fair-comparison scenario (variants 177-188), so the ten answers take ten distinct analytical
  stances rather than paraphrasing: arm-manifest symmetry auditing before any number is read;
  an explicit cost-per-token formula with fleet GPU-hour reconciliation as the falsifier;
  the calibration set treated as a hidden experimental arm with a cross-distribution sensitivity run;
  matched-SLO goodput and client-side tail latency instead of mean throughput; per-shape kernel
  attestation to exclude silent fallback to a dequantize-then-BF16 reference path; pre-registered
  decision rules and the explicit null-result statement; NCCL collective share as the Amdahl bound on
  multi-GPU decode speedup; context-length sweeps showing the KV-dominance crossover where weight-only
  quantization stops paying; structured-output and tool-call conformance as first-class gates for
  agentic consumers; and fleet heterogeneity with scheduler-enforced rollout scoping and drift
  detection. Every numeric claim in the batch is labelled ESTIMATE with an inline derivation or
  MEASURED with an artifact reference.
- Status: PROVISIONAL. These are teacher-B second-opinion rewrites produced blind, without any access
  to the teacher-A calibration directory. They are NOT expert gold, they have not been adjudicated
  against teacher-A, and they say nothing about any model's domain capability. Agreement analysis is a
  separate later step outside this worker's scope.

"""
s = s.replace("## Run 2026-08-18 batch 0188", entry + "## Run 2026-08-18 batch 0188", 1)
open(P, "w").write(s)
print("OK")
