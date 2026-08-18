import re
P="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
s=open(P).read()
entry = """## Run 2026-08-18 - train-batch-0199.jsonl

- Batch file: `results/train-batch-0199.jsonl`
- Corpus slice: `research/ai-infra-expert/corpus/train.jsonl` positional rows 1980-1989 (0-indexed), 10 records
- Source IDs: corpus-02188, corpus-02189, corpus-02190, corpus-02191, corpus-02193, corpus-02194, corpus-02195, corpus-02196, corpus-02197, corpus-02198 (note: corpus-02192 is absent from the corpus; slicing is positional, not ID arithmetic)
- Progress: 1990/2500 train (user-revised stage target of 2500, not the original 6000); remaining 510
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (`scripts/tb_verify_batch_0199.py`, derived from the 0198 verifier via sed)
- Repairs applied: none required
- Final schema check: VERIFY_PASS, aggregate TOTAL=1990, aggregate sequence confirmed a strict prefix of train.jsonl, no duplicate source_id, zero validation-batch files present
- Manifest: `MANIFEST.sha256` regenerated over all 412 files in the experiment directory (excluding MANIFEST.sha256 and __pycache__), `sha256sum -c` all-pass

Technical topics covered by this batch: all ten items are variants of the weight-only quantization fair-comparison scenario, so each answer takes a mutually exclusive analytical stance not used by any earlier batch. This round covers speculative decoding acceptance rate as a function of target precision; prefix-cache hit rate and KV block reuse changing with freed memory; power and thermal envelope with energy-per-delivered-token as the correct denominator in power-limited facilities; adapter and fine-tune compatibility against a quantized base including forced merging costs; host memory residency, NUMA staging-buffer placement and host-to-device transfer path; multi-tenant fairness and redistribution hidden by pooled throughput; unquantized-layer inventory (embeddings, output projection, scale/zero-point metadata) making realised savings smaller than the nominal bit-width ratio; queueing nonlinearity and the utilisation operating point; collective communication volume being unchanged under weight-only quantization so step speedup is bounded by compute share and overlap can be lost; and human adjudication panels treated as instruments requiring blinding, randomised order and inter-rater agreement statistics.

Every numeric statement in this batch is explicitly labelled ESTIMATE with its derivation stated, or marked as becoming MEASURED only once the named enumeration or telemetry is actually run. No teacher-A artefact was read, opened, grepped or otherwise consulted during production of this batch; the review is blind and only `source_user` / `source_assistant` from the corpus were visible.

**Status: provisional.** These outputs are teacher-B provisional review artefacts. They are not expert gold, they have not been adjudicated against teacher-A, and they say nothing about any model's domain capability.

"""
s = re.sub(r"(?m)^(## Run )", entry + r"\1", s, count=1)
open(P,"w").write(s)
print("OK", len(s))
