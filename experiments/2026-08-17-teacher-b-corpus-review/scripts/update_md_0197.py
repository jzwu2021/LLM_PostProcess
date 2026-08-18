import re
P="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
t=open(P).read()
entry = """## Run 2026-08-18 - train-batch-0197.jsonl

- Batch file: results/train-batch-0197.jsonl
- Corpus range: positional slice train.jsonl[1960:1970] (10 records)
- Source IDs: corpus-02166 .. corpus-02176 (non-consecutive; sliced by position, not ID arithmetic)
- Progress: 1970/2500 train (78.8%); remaining 530
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (scripts/tb_verify_batch_0197.py, first run, no repair needed)
- Repairs performed: none
- Final schema check: PASS (VERIFY_PASS, TOTAL 1970, strict prefix of train.jsonl confirmed, no validation-batch files)
- Manifest: regenerated MANIFEST.sha256 (403 entries); sha256sum -c all OK

Topics covered: all ten items are variants of the same weight-only quantization (WOQ)
fair-comparison prompt, so this batch attacks it from ten disjoint analytical stances not
used in any earlier batch: paged-attention block-table and allocator granularity (freed
weight bytes become capacity only in whole KV blocks, minus reservation and fragmentation);
sampler and logits-post-processing cost as the non-addressable Amdahl floor of a decode step;
continuous-batching preemption and KV-recompute accounting, with a KV-clamped arm separating
the capacity channel from the kernel channel; quantized-artefact load-path and startup
integrity (build-time and load-time hashes plus a runtime per-layer dtype dump, fail-closed);
eval-harness prompt and chat-template fidelity (identical rendered token ids across arms);
weight-load caching and repeated-read behaviour (measured HBM read volume versus assumed
one-read-per-weight, and dequantize-to-memory kernels that move more bytes than baseline);
schedule and calendar confounds (randomised interleaved A-B-B-A across distinct devices with
thermal/clock/tenant covariates); multi-model colocation and fleet packing (savings pay off
in discrete replica-per-device steps, with colocation interference); agent-loop and multi-turn
error compounding (session-level success rather than single-turn parity); and dataset/slice
freshness with certification expiry and re-calibration cost per base-model refresh.

Numeric policy: every quantitative statement in this batch is explicitly labelled ESTIMATE
with its derivation shown (block-count byte arithmetic, Amdahl over profiled buckets,
recompute waste as recomputed tokens times per-token prefill cost, artefact size from
declared scheme, measured-traffic substitution into the roofline expression, clock-to-rate
proportionality, integer packing over measured per-device capacity, independent-step
compounding, share-weighted slice aggregation). No bare numbers are presented as MEASURED.

Provisional status: these are provisional teacher-B reviews produced blind, without any
access to teacher-A outputs. They are NOT expert gold, and they do not represent or
demonstrate any model domain capability.

"""
t = t.replace("## Run 2026-08-18 - train-batch-0196.jsonl", entry + "## Run 2026-08-18 - train-batch-0196.jsonl", 1)
open(P,"w").write(t)
print("OK")
