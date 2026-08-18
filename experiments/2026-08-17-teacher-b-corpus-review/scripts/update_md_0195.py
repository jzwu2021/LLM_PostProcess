import io
P="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(P).read()
marker="## Run 2026-08-18 batch 0194"
entry = """## Run 2026-08-18 batch 0195

- Batch file: results/train-batch-0195.jsonl
- Corpus range: train.jsonl positional lines 1941-1950, ten physically consecutive rows, original
  corpus order preserved, nothing skipped or reordered. Selection by line position, not ID arithmetic.
- Source IDs: corpus-02143 through corpus-02152 (contiguous in this slice).
- Progress: 1950/2500 train records (78.0%). Remaining: 550. The 2500 denominator is the user-set
  staged target adopted on 2026-08-18, replacing the original 6000-record figure. Validation target
  is 0 for this stage; zero validation-batch files exist and none were created (verifier asserts it).
- Decisions: keep 0, rewrite 10, reject 0.
- Initial schema check: PASS on the first run of scripts/tb_verify_batch_0195.py, derived from the
  batch-0194 verifier by literal 0194->0195 substitution rather than rewritten. Checks performed:
  per-line JSON parse, batch count == 10, all 12 required fields present, teacher_lane == teacher-B,
  teacher_model == claude-opus-5-current, calibration_status == provisional, decision in
  {keep,rewrite,reject}, source_user and source_assistant character-exact against the corpus record
  (corpus schema is messages[{role,content}], flattened by role before comparison), non-empty
  corrected_answer distinct from source_assistant, ESTIMATE label present, stance marker prefix,
  quality_dimensions exactly the three integer keys in 1-5, non-empty string arrays for risks and
  evidence_required, confidence in [0,1], globally unique source_id across all 195 batches, and the
  aggregated 1950-record ID sequence being a strict prefix of train.jsonl. Verifier printed
  TOTAL 1950 then VERIFY_PASS.
- Repairs: none required. No batch was rewritten, no earlier batch or corpus file was touched.
- Final schema check: PASS (same script, unchanged output).
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself
  and excluding __pycache__; sha256sum -c reported 393 of 393 entries OK with zero mismatches.
- Topics covered by this batch: all ten rows are scenario variants 243-252 of the same
  weight-only-quantization (WOQ) serving-cost prompt, so each corrected_answer is differentiated by
  a distinct analytical stance rather than by topic. Stances in this batch, in order:
  instruction-following and format adherence (control-token logit margins, schema validity,
  agent-loop compounding); tokenizer and detokenizer invariance (conversion-tool drift in chat
  template, EOS/stop sets and generation config corrupting both quality and the cost denominator);
  speculative-decoding interaction (spec decoding and WOQ contend for the same decode bottleneck,
  acceptance-rate regression under quantization); quantization-scheme taxonomy (group size,
  symmetry, act-order, outlier retention, metadata bits-per-weight overhead); engine and dependency
  drift (kernel-selection heuristics changing on patch upgrades, recurring dual-path
  re-certification cost); energy and power envelope (joules per 1,000 output tokens, clock policy,
  power-capped racks where time savings do not convert to capacity); multi-tenant interference
  (co-tenancy, NUMA/PCIe contention, MIG/MPS, tail inflation versus mean); artefact conversion
  integrity (silently unquantized layers, loaded-config readback, per-layer reconstruction error,
  supply-chain provenance); admission control and queueing (SLO-feasible operating points,
  open- versus closed-loop generators, binding-constraint identification); and regression triage
  and attribution (instrumentation designed so a failure is diagnosable without a re-run).
  Every numeric claim in every answer is labelled ESTIMATE with its derivation shown inline.
- Status of this output: PROVISIONAL teacher-B review under blind conditions. No file under
  experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or searched during this
  run, so no anchoring on teacher-A corrected answers is possible. These records are NOT expert gold
  labels, have not been human-verified, and say nothing about any model's domain capability. They
  are one model's independent second opinion, to be used only as input to a later, separate
  agreement analysis.

"""
assert marker in s
s=s.replace(marker, entry+marker, 1)
open(P,"w").write(s)
print("OK")
