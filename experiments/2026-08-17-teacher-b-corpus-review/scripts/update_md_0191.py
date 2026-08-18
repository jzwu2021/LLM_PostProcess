P = '/media/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md'
s = open(P).read()
ANCHOR = "## Run 2026-08-18 batch 0190"
assert ANCHOR in s
entry = """## Run 2026-08-18 batch 0191

- Batch file: results/train-batch-0191.jsonl
- Corpus range: train.jsonl positional lines 1901-1910, ten physically consecutive rows, original
  corpus order preserved, nothing skipped or reordered. Selection by line position, not ID arithmetic
  (the window contains an ID gap: corpus-02108 is absent from the corpus).
- Source IDs: corpus-02099, 02100, 02101, 02102, 02103, 02104, 02105, 02106, 02107, 02109.
- Progress: 1910/2500 train records (76.4%). Remaining: 590. The 2500 denominator is the user-set
  staged target adopted on 2026-08-18, replacing the original 6000-record figure. Validation target
  is 0 for this stage; zero validation-batch files exist and none were created (verifier asserts it).
- Decisions: keep 0, rewrite 10, reject 0.
- Initial schema check: PASS on the first run of scripts/tb_verify_batch_0191.py, derived from the
  batch-0190 verifier by literal 0190->0191 substitution rather than rewritten. Checks: per-line JSON
  parse, batch count 10, all 12 required fields, fixed values for teacher_lane / teacher_model /
  calibration_status, decision in {keep,rewrite,reject}, byte-exact equality of source_user and
  source_assistant against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer,
  confidence in [0,1], distinct answer openings, global source_id uniqueness across all batches, and
  the aggregate train sequence being a strict prefix of train.jsonl (1910 records).
- Repair actions: none required this run.
- Final schema check: PASS (VERIFY_PASS).
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself;
  sha256sum -c reports all files OK.
- Technical topics covered: the ten prompts continue the weight-only quantization fair-comparison
  scenario (variants 199-209), so the ten answers again take ten distinct analytical stances, disjoint
  from those used in batch 0190: kernel dispatch attestation, distinguishing a loaded quantized
  checkpoint from an actually-executed low-precision kernel and bounding the compute claim by the
  traffic-weighted fraction of GEMM time that avoided a silent dequantize-to-BF16 fallback; roofline
  and bottleneck attribution separating memory-bandwidth-bound decode from compute-bound prefill and
  deriving the end-to-end projection by time-share weighting; calibration-set validity as a controlled
  variable, with disjoint-draw checkpoint rebuilds, cross-draw spread reporting and an explicit
  calibration-evaluation contamination check; tail-latency and percentile SLO framing, including the
  mechanism by which freed memory admits larger batches whose long prefill steps damage decode
  inter-token latency even as mean throughput improves; structured-output and tool-call conformance as
  a discrete axis measured by parser and schema validation over retained raw generations, including
  the way constrained decoding converts syntactic failure into harder-to-detect argument-value error;
  an explicit cost-per-million-output-tokens formula at fixed SLO attainment with amortised build,
  calibration, qualification and re-qualification cost over a stated artefact lifetime; confounder
  enumeration via mechanically diffed arm manifests down to node identity, with a third BF16 arm at
  matched batch settings when freed memory is the real mechanism; multi-GPU interaction, where
  collectives still move compute-dtype activations so the end-to-end gain decays as tensor-parallel
  degree rises, plus quantization-group to shard-boundary alignment and cross-rank numerical
  consistency; long-horizon drift with a retained BF16 shadow, soak monitoring and a re-qualification
  trigger list; and decision-rule governance requiring a timestamped pre-registered rule, an
  independent reviewer, recorded deviations and an explicitly scoped verdict. Every numeric claim in
  the batch is labelled ESTIMATE with an inline derivation or MEASURED with an artifact reference.
- Status: PROVISIONAL. These are teacher-B second-opinion rewrites produced blind, without any access
  to the teacher-A calibration directory. They are NOT expert gold, they have not been adjudicated
  against teacher-A, and they say nothing about any model's domain capability. Agreement analysis is a
  separate later step outside this worker's scope.

"""
s = s.replace(ANCHOR, entry + ANCHOR, 1)
open(P, 'w').write(s)
print("MD_UPDATED")
