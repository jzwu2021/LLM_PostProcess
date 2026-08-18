P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
BLOCK = """## Round 209 - train-batch-0209.jsonl

- Batch file: results/train-batch-0209.jsonl
- Corpus slice: train.jsonl positional lines 2081-2090 (0-indexed 2080-2089)
- Source IDs: corpus-02292 .. corpus-02302 (non-consecutive: corpus-02297 is absent from the corpus, so the slice is taken positionally, never by ID arithmetic)
- Progress: 2090/2500 train (410 remaining); validation target is 0 by user decision
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run (tb_verify_batch_0209.py, derived from the 0208 verifier via sed on batch number and slice offset)
- Repairs performed: none required
- Final schema check: VERIFY_PASS, aggregate TOTAL=2090, strict prefix of train.jsonl confirmed, no validation-batch files present
- Manifest: MANIFEST.sha256 regenerated over all files in the experiment directory except itself; sha256sum -c passed for every entry

Technical topics covered by this batch: all ten items belong to the same degenerate TP-versus-PP family (scenario variants 92-102) whose assistant turn is a grading rubric rather than an answer, so every item is a rewrite. Ten mutually disjoint analytical stances (Stance 90-99) were used to keep a near-homogeneous batch from producing homogeneous output: weight/KV quantisation moving the memory floor and therefore the minimum feasible tensor degree; prefix and KV-cache reuse collapsing the prefill-borne share of the inter-layout gap, plus the requirement that cached KV match the serving layout's sharding; prefill/decode disaggregation letting each phase take its own layout at the cost of a per-request KV transfer on the TTFT path; multi-tenancy and noisy-neighbour contention on host CPU, PCIe and interconnect reordering the arms; autoscaling, cold-start and capacity-restoration time as an availability constraint set by devices per replica; failure-domain arithmetic where replica availability is per-device availability raised to the group size; scheduler and batching policy (continuous batching, chunked prefill, admission and preemption) as confounders large enough to swap the ranking when engine defaults are used; statistical design, interleaved repeats and confidence intervals on tail percentiles; cost per SLO-sustaining request as the decision variable once both arms meet the SLO; and a closing provenance/authority-bound stance.

Every quantitative claim in this batch is explicitly tagged ESTIMATE with its derivation attached; no value is MEASURED, because no benchmark was executed for this review. These records are provisional teacher-B review material, NOT expert gold, and they are not evidence about any model's domain capability. Teacher-A artifacts were not read at any point during generation (blind review preserved).

"""
c = open(P).read()
i = c.index("## Round 208")
open(P, "w").write(c[:i] + BLOCK + c[i:])
print("ok")
