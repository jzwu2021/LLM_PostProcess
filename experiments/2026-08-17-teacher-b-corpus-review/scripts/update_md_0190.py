P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s = open(P).read()
entry = """## Run 2026-08-18 batch 0190

- Batch file: results/train-batch-0190.jsonl
- Corpus range: train.jsonl positional lines 1891-1900, ten physically consecutive rows, original
  corpus order preserved, nothing skipped or reordered. Selection by line position, not ID arithmetic.
- Source IDs: corpus-02089, 02090, 02091, 02092, 02093, 02094, 02095, 02096, 02097, 02098.
- Progress: 1900/2500 train records (76.0%). Remaining: 600. The 2500 denominator is the user-set
  staged target adopted on 2026-08-18, replacing the original 6000-record figure. Validation target
  is 0 for this stage; zero validation-batch files exist and none were created (verifier asserts it).
- Decisions: keep 0, rewrite 10, reject 0.
- Initial schema check: PASS on the first run of scripts/tb_verify_batch_0190.py, derived from the
  batch-0189 verifier by literal 0189->0190 substitution rather than rewritten. Checks: per-line JSON
  parse, batch count 10, all 12 required fields, fixed values for teacher_lane / teacher_model /
  calibration_status, decision in {keep,rewrite,reject}, byte-exact equality of source_user and
  source_assistant against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer,
  confidence in [0,1], distinct answer openings, global source_id uniqueness across all batches, and
  the aggregate train sequence being a strict prefix of train.jsonl (1900 records).
- Repair actions: none required this run.
- Final schema check: PASS (VERIFY_PASS).
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself;
  sha256sum -c reports all files OK.
- Technical topics covered: the ten prompts continue the weight-only quantization fair-comparison
  scenario (variants 189-198), so the ten answers again take ten distinct analytical stances rather
  than paraphrasing, and they are disjoint from the stances used in batch 0189: per-slice quality
  floors with bootstrap CIs and an A/A noise baseline instead of aggregate scores; explicit device
  memory byte accounting showing whether freed weight bytes actually become usable KV blocks rather
  than being reabsorbed by scales, workspace and fragmentation; statistical power, paired-difference
  analysis and pre-registered stopping rules separating 'no difference measured' from 'equivalence
  established'; warmup discarding and stationarity testing against autotuning, allocator growth and
  sustained-load clock drop; the numerical mechanism itself - group dynamic range over grid levels,
  outlier channels, and a selective-BF16-masking arm as both localisation and mitigation; operational
  reversibility with timed rollback rehearsal, canary abort conditions and mixed-fleet consistency
  during rollout; batch-composition realism via production trace replay preserving arrival timing,
  with prefill-decode split and preemption counts; opportunity cost against substitute levers such as
  KV quantization, prefix caching, scheduling and speculative decoding measured on the same harness;
  reproducibility as a published artefact bundle with a named independent reproduction owner and a
  pre-agreed reproduction band; and supply chain, licence and calibration-data governance including
  byte-reproducible rebuild of the quantized checkpoint from recorded inputs. Every numeric claim in
  the batch is labelled ESTIMATE with an inline derivation or MEASURED with an artifact reference.
- Status: PROVISIONAL. These are teacher-B second-opinion rewrites produced blind, without any access
  to the teacher-A calibration directory. They are NOT expert gold, they have not been adjudicated
  against teacher-A, and they say nothing about any model's domain capability. Agreement analysis is a
  separate later step outside this worker's scope.

"""
s = s.replace("## Run 2026-08-18 batch 0189", entry + "## Run 2026-08-18 batch 0189", 1)
open(P, "w").write(s)
print("OK")
