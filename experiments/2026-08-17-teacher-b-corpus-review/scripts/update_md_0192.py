P = '/media/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md'
s = open(P).read()
ANCHOR = "## Run 2026-08-18 batch 0191"
assert ANCHOR in s
entry = """## Run 2026-08-18 batch 0192

- Batch file: results/train-batch-0192.jsonl
- Corpus range: train.jsonl positional lines 1911-1920, ten physically consecutive rows, original
  corpus order preserved, nothing skipped or reordered. Selection by line position, not ID arithmetic.
- Source IDs: corpus-02111, 02112, 02113, 02114, 02115, 02116, 02117, 02118, 02119, 02120.
- Progress: 1920/2500 train records (76.8%). Remaining: 580. The 2500 denominator is the user-set
  staged target adopted on 2026-08-18, replacing the original 6000-record figure. Validation target
  is 0 for this stage; zero validation-batch files exist and none were created (verifier asserts it).
- Decisions: keep 0, rewrite 10, reject 0.
- Initial schema check: PASS on the first run of scripts/tb_verify_batch_0192.py, derived from the
  batch-0191 verifier by literal 0191->0192 substitution rather than rewritten. Checks: per-line JSON
  parse, batch count 10, all 12 required fields, fixed values for teacher_lane / teacher_model /
  calibration_status, decision in {keep,rewrite,reject}, byte-exact equality of source_user and
  source_assistant against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer,
  confidence in [0,1], distinct answer openings, global source_id uniqueness across all batches, and
  the aggregate train sequence being a strict prefix of train.jsonl (1920 records).
- Repair actions: none required this run.
- Final schema check: PASS (VERIFY_PASS).
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself
  (380 files); sha256sum -c reports all files OK.
- Technical topics covered: the ten prompts continue the weight-only quantization fair-comparison
  scenario (variants 211-220), so the ten answers again take ten distinct analytical stances, disjoint
  from those used in batch 0191: device-level memory byte accounting reconciling weights, per-group
  scales and zero points, workspace, CUDA graph buffers and fragmentation against measured free bytes
  and converting the remainder into an admitted-concurrency figure via per-token KV cost; the
  numerical outlier mechanism, localising damage by per-layer and per-channel error against a BF16
  reference and correlating it with activation range statistics that drive scale selection;
  statistical power and the A/A noise floor, requiring the measured delta to exceed harness variation
  and sizing evaluation items and repetitions in advance; traffic representativeness, replaying
  production traces to preserve the joint length distribution, arrival burstiness, concurrency and
  prompt-cache hit rate because the gain scales with decode share; rollout mechanics with a rehearsed
  rollback, warm reference artefact, dual-arm capacity headroom and enumeration of non-reverting state
  such as cached outputs; reproducibility and artefact identity pinned by content hash across
  checkpoints, calibration set, quantization tool, engine, kernel library, driver, container digest,
  SKU, parallel degree and seeds, with legitimate nondeterminism documented; the alternative-hypothesis
  comparison costing a smaller base model, scheduler and batching tuning applied to the baseline first,
  prefix and prompt caching, speculative decoding and a different accelerator SKU on the same
  cost-per-token metric; fairness and safety regression per pre-registered slice including refusal rate
  and safety-classifier agreement with an independent verdict owner; measurement hygiene covering
  discarded warmup, sub-window stationarity, repetition across independent process launches, a verified
  non-saturated load generator and interleaved arm ordering; and a written, machine-readable
  generalisation envelope enforcing that the verdict does not transfer across hardware generation,
  parallel degree, context length or traffic mix. Every numeric statement is labelled ESTIMATE with its
  derivation stated; no MEASURED value is asserted, since no measurement was performed here.
- Status: provisional teacher-B output. It is NOT expert gold, has not been adjudicated against
  teacher-A, and says nothing about any model's domain capability. It records one blind reviewer pass.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was
  read, opened, grepped or listed during this run.

"""
s = s.replace(ANCHOR, entry + ANCHOR, 1)
open(P, "w").write(s)
print("MD_UPDATED")
