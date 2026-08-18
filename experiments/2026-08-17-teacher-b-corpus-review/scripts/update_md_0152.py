import io, os

P = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/EXPERIMENT.md"
ANCHOR = "## Run 2026-08-18 batch 0151"

NEW = """## Run 2026-08-18 batch 0152

- Batch file: results/train-batch-0152.jsonl
- Corpus range: train.jsonl lines 1511-1520 (0-indexed 1510-1519)
- Source IDs: corpus-01668 through corpus-01678 (contiguous in corpus order; corpus-01670 is not present in train.jsonl, IDs are not dense)
- Progress: train 1520/5399, validation 0/601, total 1520/6000, remaining 4480
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS on first run of scripts/adhoc_verify_batch_0152.py (no repair actions were needed)
- Repair actions: none
- Final schema check: PASS - 10 records, all 12 required fields present, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, source_user/source_assistant byte-identical to research/ai-infra-expert/corpus/train.jsonl, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all 152 batches, train aggregate is a strict prefix of train.jsonl, validation aggregate empty and therefore a trivial prefix, train batch numbering contiguous 0001-0152
- Manifest: MANIFEST.sha256 regenerated over every file in this experiment directory except MANIFEST.sha256 itself; sha256sum -c reports OK for all entries
- Verification independence: scripts/adhoc_verify_batch_0152.py was written fresh for this run rather than reusing the previous batch verifier, so the generator and the checker do not share code

Technical topics covered by this batch: all ten items are hard design_or_diagnosis prompts on multi-GPU collective-initialization hangs (scenario variants 68-78) spread across Troubleshooting, Performance Analysis, and System Design categories. The rewritten answers decompose init into rendezvous (TCPStore on MASTER_ADDR:MASTER_PORT), ncclUniqueId exchange, and bootstrap/transport negotiation (NVLink/P2P, SHM, NET over IB/RoCE or sockets), and each gives a distinct falsifiable hypothesis with a controlled experiment: gloo-vs-NCCL phase isolation, serialized TCP connect timeouts and NCCL_SOCKET_IFNAME pinning, node-pair bisection with NCCL_P2P_DISABLE as a diagnostic (never a fix), nccl-tests busbw compared against the path's theoretical bound to detect silent NVLink->PCIe or RDMA->TCP fallback, bounded init timeout plus a pre-flight all_reduce as a design-level observability change, rank-inventory arithmetic from /proc/<pid>/environ, timestamp-delta histograms over per-rank NCCL_DEBUG_FILE logs, control-plane vs data-plane interface separation with RoCE GID/PFC caveats, py-spy/gdb stack classification across ranks, and A/B image comparison for driver/NCCL/peermem version skew. Every answer carries explicit confounders, required evidence, and a rollback gate (one variable per trial, NCCL_ASYNC_ERROR_HANDLING=1, revert within two runs).

Status caveat: these outputs are PROVISIONAL teacher-B second opinions produced under blind review. They are not expert gold labels, they have not been validated against a live cluster, and they say nothing about any model's domain capability. This lane was produced without reading any teacher-A artifact.

"""

src = open(P, encoding="utf-8").read()
assert ANCHOR in src, "anchor not found"
assert src.count(ANCHOR) == 1, "anchor not unique"
assert "## Run 2026-08-18 batch 0152" not in src, "batch 0152 section already present"
out = src.replace(ANCHOR, NEW + ANCHOR, 1)
open(P, "w", encoding="utf-8").write(out)
print("inserted; bytes", len(src), "->", len(out))
