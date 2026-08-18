import re
P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s = open(P).read()
entry = """## Run 2026-08-18 - train-batch-0198.jsonl

- Batch file: results/train-batch-0198.jsonl
- Corpus range: positional slice train.jsonl[1970:1980] (10 records)
- Source IDs: corpus-02178 .. corpus-02187 (sliced by position, not ID arithmetic)
- Progress: 1980/2500 train (79.2%); remaining 520
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (scripts/tb_verify_batch_0198.py, first run, no repair needed)
- Repairs performed: none
- Final schema check: PASS (VERIFY_PASS, TOTAL 1980, strict prefix of train.jsonl confirmed, no validation-batch files)
- Manifest: regenerated MANIFEST.sha256 (408 entries); sha256sum -c all OK

Topics covered: all ten items are variants of the same weight-only quantization (WOQ)
fair-comparison prompt, so this batch attacks it from ten disjoint analytical stances not
used in any earlier batch: KV-cache dtype as a second precision axis that presets silently
couple to weights, with context-length-dependent error accumulation distinct from stationary
weight error; calibration-set provenance and leakage between the calibration corpus and the
evaluation slices, plus calibration-induced artefact non-reproducibility; quantized kernel
dispatch and shape coverage, where fast-path constraints on batch and alignment make the
traffic-weighted speedup diverge from the peak speedup; numerical determinism and same-arm
run-to-run variance under continuous batching as the precondition for any cross-arm delta;
request cancellation and client disconnect, introducing goodput and cancelled-request KV
block-hours against run-to-completion throughput; observability and metric-pipeline integrity
(histogram bucketing, percentile merge validity, sampled tracing uncertainty, counter resets,
clock skew); build and conversion cost amortisation with a payback period compared against the
organisation's base-model refresh cadence; cold-start and autoscaling behaviour, where load,
conversion and autotune phases set the standing headroom an elastic fleet must hold; failure
and degradation-mode parity under fault injection, including silent numerical degradation that
liveness probes cannot detect; and decision authority and conflict of interest, requiring
pre-registration timestamped before first measurement and sign-off separated from the runner.

These results are provisional teacher-B output produced blind (teacher-A artefacts were not
read at any point). They are NOT expert gold, have not been human-adjudicated, and do not
represent any model domain capability claim.

"""
s = s.replace("## Run 2026-08-18 - train-batch-0197.jsonl", entry + "## Run 2026-08-18 - train-batch-0197.jsonl", 1)
open(P, "w").write(s)
print("OK")
