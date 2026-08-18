P = '/media/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md'
s = open(P).read()
ANCHOR = "## Run 2026-08-18 batch 0193"
assert ANCHOR in s
entry = """## Run 2026-08-18 batch 0194

- Batch file: results/train-batch-0194.jsonl
- Corpus range: train.jsonl positional lines 1931-1940, ten physically consecutive rows, original
  corpus order preserved, nothing skipped or reordered. Selection by line position, not ID arithmetic.
- Source IDs: corpus-02131, 02133, 02134, 02135, 02136, 02137, 02138, 02139, 02141, 02142
  (corpus IDs are non-consecutive here; positional slicing is what guarantees prefix integrity).
- Progress: 1940/2500 train records (77.6%). Remaining: 560. The 2500 denominator is the user-set
  staged target adopted on 2026-08-18, replacing the original 6000-record figure. Validation target
  is 0 for this stage; zero validation-batch files exist and none were created (verifier asserts it).
- Decisions: keep 0, rewrite 10, reject 0.
- Initial schema check: PASS on the first run of scripts/tb_verify_batch_0194.py, derived from the
  batch-0193 verifier by literal 0193->0194 substitution rather than rewritten. Checks: per-line JSON
  parse, batch count 10, all 12 required fields, fixed values for teacher_lane / teacher_model /
  calibration_status, decision in {keep,rewrite,reject}, byte-exact equality of source_user and
  source_assistant against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer,
  confidence in [0,1], distinct answer openings, global source_id uniqueness across all batches, and
  the aggregate train sequence being a strict prefix of train.jsonl (1940 records).
- Repair actions: none required this run.
- Final schema check: PASS (VERIFY_PASS).
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself
  (390 files, __pycache__ excluded); sha256sum -c reports all 390 files OK.
- Technical topics covered: the ten prompts continue the weight-only quantization fair-comparison
  scenario (variants 231-242), so the ten answers take ten analytical stances disjoint from every
  stance used in batches 0181-0193: outlier-channel and activation-distribution structure, requiring
  per-channel magnitude dumps and per-group dynamic-range correlation with layer output error before
  a uniform scheme is compared against outlier-aware variants and group-size choices; determinism and
  reproducibility, requiring a full artefact-and-environment bundle reproduced by an independent
  operator plus a measured A/A noise floor that sets the minimum detectable effect; warmup versus
  steady state, discarding transients that differ between arms because of autotuning and allocator
  behaviour, and reporting cold-start readiness separately since it governs autoscaling and recovery;
  request-trace realism, validating input and output length tails, arrival burstiness, prefix-sharing
  structure and request-class mix against production telemetry and replaying with production timing so
  queueing is measured rather than eliminated; hardware and numeric support, stating that weight-only
  schemes are bandwidth-side only where arithmetic still runs at BF16 rates, requiring measured
  achieved throughput on the deployed device generation and per-generation verdicts on heterogeneous
  fleets; rollout mechanics and reversibility, requiring an exercised rollback drill under load with
  the baseline artefact kept warm and detection plus decision plus execution latency summed as the
  exposure window; statistical power and slice multiplicity, noting the asymmetry that an underpowered
  test systematically favours adoption because undetected harm reads as absence of harm; baseline
  tuning parity, treating tuning effort as a recorded variable because within-arm configuration spread
  can rival the between-arm delta; memory accounting and fragmentation, tracing freed weight bytes
  through allocator reserve, activation peaks, collective workspace and burst headroom to measured
  sustained concurrency rather than computing capacity from checkpoint sizes; and governance,
  provenance and supply chain, requiring base checkpoint licence, quantization tool version,
  calibration data rights, build digest and load-time integrity verification, with third-party
  pre-quantized checkpoints flagged as substituting another party's unaudited choices. Every numeric
  statement is labelled ESTIMATE with its derivation stated; no MEASURED value is asserted, since no
  measurement was performed here.
- Status: provisional teacher-B output. It is NOT expert gold, has not been adjudicated against
  teacher-A, and says nothing about any model's domain capability. It records one blind reviewer pass.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was
  read, opened, grepped or listed during this run.

"""
s = s.replace(ANCHOR, entry + ANCHOR, 1)
open(P, "w").write(s)
print("MD_UPDATED")
