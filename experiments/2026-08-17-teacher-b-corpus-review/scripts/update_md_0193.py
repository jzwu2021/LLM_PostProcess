P = '/media/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md'
s = open(P).read()
ANCHOR = "## Run 2026-08-18 batch 0192"
assert ANCHOR in s
entry = """## Run 2026-08-18 batch 0193

- Batch file: results/train-batch-0193.jsonl
- Corpus range: train.jsonl positional lines 1921-1930, ten physically consecutive rows, original
  corpus order preserved, nothing skipped or reordered. Selection by line position, not ID arithmetic.
- Source IDs: corpus-02121, 02122, 02123, 02124, 02125, 02126, 02127, 02128, 02129, 02130.
- Progress: 1930/2500 train records (77.2%). Remaining: 570. The 2500 denominator is the user-set
  staged target adopted on 2026-08-18, replacing the original 6000-record figure. Validation target
  is 0 for this stage; zero validation-batch files exist and none were created (verifier asserts it).
- Decisions: keep 0, rewrite 10, reject 0.
- Initial schema check: PASS on the first run of scripts/tb_verify_batch_0193.py, derived from the
  batch-0192 verifier by literal 0192->0193 substitution rather than rewritten. Checks: per-line JSON
  parse, batch count 10, all 12 required fields, fixed values for teacher_lane / teacher_model /
  calibration_status, decision in {keep,rewrite,reject}, byte-exact equality of source_user and
  source_assistant against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer,
  confidence in [0,1], distinct answer openings, global source_id uniqueness across all batches, and
  the aggregate train sequence being a strict prefix of train.jsonl (1930 records).
- Repair actions: none required this run.
- Final schema check: PASS (VERIFY_PASS).
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself
  (385 files); sha256sum -c reports all 385 files OK.
- Technical topics covered: the ten prompts continue the weight-only quantization fair-comparison
  scenario (variants 221-230), so the ten answers take ten analytical stances disjoint from every
  stance used in batches 0181-0192: kernel dispatch and the dequantization path, requiring
  kernel-level profiling to prove a fused low-precision kernel runs rather than a
  dequantize-then-BF16-GEMM fallback that keeps the memory saving while restoring full-precision
  bandwidth; batch-size regime and roofline position, sweeping concurrency to show the advantage
  collapsing from the bandwidth-bound to the compute-bound regime and noting that freed memory raises
  admissible batch size and therefore partially cancels the latency win; calibration-data provenance,
  treating the quantized checkpoint as a fit to a distribution, requiring disjointness from the
  evaluation set and a sensitivity test across at least two calibration corpora; decision-rule
  pre-registration with timestamped thresholds, per-slice vetoes, a declared inconclusive default and
  an independent decision owner; cost-model completeness, adding qualification effort, recurring
  re-qualification on engine and driver upgrades, dual-artefact storage and rollback headroom to the
  per-token figure at fixed SLO attainment; long-context and KV share, showing the weight-byte benefit
  declining as KV bytes dominate and redirecting alternatives toward KV-cache quantization and
  attention variants; failure modes and blast radius, exercising numerical overflow, long-generation
  error accumulation, repetition-driven output-length inflation, burst OOM with a larger KV pool and
  rare-shape kernel failures with detection latency and containment compared against the baseline;
  post-adoption observability, instrumenting the gating metrics themselves with precision as a metric
  label and a BF16 shadow-probe agreement check; parallelism interaction, showing the addressable
  step-time share shrinking as the collective's share grows with tensor-parallel degree and requiring
  quantization group boundaries to be compatible with each weight matrix's sharding; and evaluation
  instrument validity, requiring a known-bad positive control to separate beyond the A/A noise floor
  before a parity result is treated as informative, with a pinned and human-agreement-validated judge
  where an LLM judge is used. Every numeric statement is labelled ESTIMATE with its derivation stated;
  no MEASURED value is asserted, since no measurement was performed here.
- Status: provisional teacher-B output. It is NOT expert gold, has not been adjudicated against
  teacher-A, and says nothing about any model's domain capability. It records one blind reviewer pass.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was
  read, opened, grepped or listed during this run.

"""
s = s.replace(ANCHOR, entry + ANCHOR, 1)
open(P, "w").write(s)
print("MD_UPDATED")
