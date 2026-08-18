import re
P="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(P).read()
entry = """## Run 2026-08-18 - train-batch-0200.jsonl

- Batch file: `results/train-batch-0200.jsonl`
- Corpus slice: `research/ai-infra-expert/corpus/train.jsonl` positional rows 1990-1999 (0-indexed), 10 records
- Source IDs: corpus-02199, corpus-02200, corpus-02201, corpus-02202, corpus-02203, corpus-02204, corpus-02205, corpus-02206, corpus-02207, corpus-02208 (slicing is positional, not ID arithmetic)
- Progress: 2000/2500 train (user-revised stage target of 2500, not the original 6000); remaining 500
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (`scripts/tb_verify_batch_0200.py`, derived from the 0199 verifier via sed)
- Repairs applied: none required
- Final schema check: VERIFY_PASS, aggregate TOTAL=2000, aggregate sequence confirmed a strict prefix of train.jsonl, no duplicate source_id, zero validation-batch files present
- Manifest: `MANIFEST.sha256` regenerated over all files in the experiment directory (excluding MANIFEST.sha256 and __pycache__), `sha256sum -c` all-pass

Technical topics covered by this batch: this batch spans a scenario boundary. The first two items (corpus-02199, corpus-02200) close out the weight-only quantization fair-comparison scenario with two remaining mutually exclusive stances: vendor support-matrix and attestation coverage of the quantized serving path, including licensing of re-derived checkpoints and the cost of self-supporting unsupported kernels; and eval-suite staleness, covering resolution decay near the score ceiling, prompt-distribution drift away from production traffic, and hash-level disjointness between calibration and eval sets. The remaining eight items (corpus-02201 through corpus-02208) open a new scenario, tensor versus pipeline parallelism for a latency-sensitive multi-GPU service, and are given their own shared substrate rather than reusing the quantization one. Stances there: latency-metric decomposition, showing TTFT and TPOT are governed by different bottlenecks so a single collapsed number can select the wrong layout; pipeline bubble arithmetic, with the (S-1)/(M+S-1) idle fraction approaching (S-1)/S at concurrency one; collective topology and node-boundary containment of tensor-parallel groups, including small-message latency-bound collectives at decode-step sizes; KV cache capacity as the concurrency ceiling, manifesting as preemption and recompute rather than out-of-memory; fault domain and reconfiguration cost, with blast radius differing between group-fatal TP and stage-scoped PP; layer-partition balance against heterogeneous block cost, where embedding and output-projection stages break equal-count partitions; hybrid layout search over the full (TP, PP, data-parallel) factorisation of the device count, treating the posed binary as a false dichotomy that discards dominating interior points; and numerical-equivalence testing as a correctness gate, distinguishing benign floating-point reassociation from sharding or collective defects via first-position logit comparison under greedy decoding.

Every numeric statement in this batch is explicitly labelled ESTIMATE with its derivation stated, or marked as becoming MEASURED only once the named enumeration, profile or telemetry is actually run. No teacher-A artefact was read, opened, grepped or otherwise consulted during production of this batch; the review is blind and only `source_user` / `source_assistant` from the corpus were visible.

**Status: provisional.** These outputs are teacher-B provisional review artefacts. They are not expert gold, they have not been adjudicated against teacher-A, and they say nothing about any model's domain capability.

"""
s = re.sub(r"(?m)^(## Run )", entry + r"\1", s, count=1)
open(P,"w").write(s)
print("OK", len(s))
