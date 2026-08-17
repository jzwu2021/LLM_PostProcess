import json, os

CORPUS = 'research/ai-infra-expert/corpus/train.jsonl'
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0029.jsonl'
START, N = 280, 10

A = {}

A[280] = """Contrast: MoE FFN vs the naive dense FFN, framed as a serving decision.

Naive baseline. One FFN, every token pays d_model x d_ff twice (up and down projection),
~2 x 2 x d_model x d_ff FLOPs per token. Capacity is bought only by widening d_ff, and the
cost is paid by every token on every step. In exchange you get a fully static compute graph:
no router, no capacity factor, no token dropping, no all-to-all.

MoE. N experts + a router; top-k routing means parameters scale with N while per-token FLOPs
scale with k. For equal quality an MoE typically needs far fewer active FLOPs than the dense
model it matches, which is why it wins on training compute and on large-batch prefill.

Concrete mechanism: the capacity factor. Each expert on each rank is given a fixed buffer of
capacity_factor x (tokens_per_rank x k / N) slots. Tokens routed to an already-full expert are
dropped (or bounced to the residual path). That fixed buffer is what makes the all-to-all a
static-shape collective and therefore fusable and CUDA-graph-friendly; it is also exactly what
converts routing skew into silent quality loss rather than into an OOM. Instrument drop rate
per layer - it is the single most informative MoE health metric and it is not visible in loss
until it is already large.

Boundary condition: the advantage inverts in memory-bound decode. At batch=1 decode you read
k expert weight matrices per layer per token instead of one, so HBM traffic per token goes up
by roughly k x (expert_params / dense_params) even though FLOPs went down. An MoE that is
2x cheaper in training FLOPs can be slower per token in low-concurrency decode than the dense
model. The crossover is set by arithmetic intensity, not by parameter count.

Falsifiable claim and evidence. Claim: on this cluster, MoE decode throughput exceeds dense
only above some concurrency C*. Test: sweep concurrency 1,4,16,64,256 at fixed input/output
lengths, record tokens/s and per-layer drop rate; C* is where the curves cross. Rollback gate:
if measured p95 TTFT or TPOT regresses beyond the SLO at the target concurrency, or drop rate
exceeds ~1% at the shipped capacity factor, revert to the dense checkpoint.

Assumptions: transformer decoder, top-k token-choice routing, expert parallelism across ranks.
No vendor- or version-specific numbers are asserted here; all thresholds must be measured."""

A[281] = """Two failure modes / trade-offs of MoE.

1) Routing collapse and load imbalance. The router is trained jointly with the experts and has
a self-reinforcing incentive: an expert that receives more tokens gets more gradient, becomes
better, and is then chosen more often. Left unchecked, effective expert count falls far below N
and you pay N-experts of memory for a handful of useful ones.

Concrete mechanism: the auxiliary load-balancing loss. It penalises the dot product of the
fraction of tokens routed to each expert with the mean router probability for that expert,
pushing the routing distribution toward uniform. Its coefficient is a real trade-off knob - too
small and you get collapse, too large and it fights the language-modelling objective and costs
quality. Some newer designs replace it with a per-expert bias term adjusted from observed load,
which balances without injecting gradient noise into the main loss.

2) All-to-all as the new critical path. Under expert parallelism each MoE layer performs two
all-to-all collectives, dispatch and combine. All-to-all is bisection-bandwidth- and
latency-bound, unlike the all-reduce a dense TP layer uses, and it does not overlap for free -
it sits between the attention block and the expert GEMM. Cross-node MoE over RoCE with an
unconfigured or mistuned lossless fabric (PFC/ECN thresholds wrong) shows up as long-tail
step-time jitter, not as an error.

Boundary condition: both failure modes are load-dependent and only appear at scale. Balance
looks fine on a small single-node smoke run because tokens_per_rank is large relative to the
capacity buffer; skew bites when you shard experts across nodes and per-rank token counts drop.
Likewise all-to-all is invisible while the expert group fits inside one NVLink domain and
becomes dominant the moment it crosses the node boundary.

Evidence required: per-layer and per-expert token counts (Gini or max/mean load ratio), token
drop rate at the shipped capacity factor, and a profile split of step time into attention,
expert GEMM, and all-to-all. Rollback gate: max/mean expert load above ~2x sustained, or
all-to-all exceeding ~25% of step time, means stop and fix topology or routing before scaling
further."""

A[282] = """Two failure modes / trade-offs of MoE.

1) Token dropping under a fixed capacity factor. Capacity per expert is preallocated so the
dispatch collective has a static shape. Tokens beyond that buffer are dropped and pass through
the residual only. This is a silent failure: no exception, no OOM, just a slow quality tax that
correlates with sequence content. Domain-shifted or long-context batches route more unevenly
than the training mix, so drop rate is highest exactly on the traffic you care about.

Concrete mechanism: dropless / variable-capacity implementations (grouped GEMM over
ragged expert batches) remove the drop at the cost of dynamic shapes, which breaks CUDA graph
capture and makes step time input-dependent. That is the trade: deterministic latency versus
guaranteed token coverage. Pick one deliberately, per deployment.

2) Memory and checkpoint cost scale with total, not active, parameters. An MoE with N experts
holds N copies of the FFN weights plus optimizer state. In training with Adam that is roughly
weights + 2 moments + master copy, so total-parameter growth multiplies straight through into
HBM and into checkpoint size and save/restore wall time. Expert parallelism spreads it, but it
also fixes your minimum world size: you cannot serve the model on fewer GPUs than the weights
require, no matter how small your traffic is. Small-scale evaluation and canarying get
expensive.

Boundary condition: the memory argument dominates at serving time, the drop argument dominates
at training time. If HBM per GPU is the binding constraint, an MoE is worse than a dense model
of equal active FLOPs; if training compute is the binding constraint, it is better. On 24GB-class
accelerators the MoE weight footprint usually decides the question before quality does.

Evidence required: measured per-layer drop rate on production-like traffic; measured resident
HBM for weights vs KV cache vs activations; checkpoint save/restore timing. Falsifiable claim:
raising capacity factor from 1.0 to 1.25 reduces drop rate by more than it costs in step time -
test by sweeping and measuring both. Rollback gate: if the memory headroom for KV cache falls
below the level needed for the target concurrency, the MoE config is not shippable."""

A[283] = """Two failure modes / trade-offs of MoE.

1) Router instability during training. The router output is an argmax over a softmax; the
argmax is discontinuous, so a tiny logit change flips a token to a different expert and produces
a large loss change. Symptoms are loss spikes, poor reproducibility across restarts, and
sensitivity to batch composition. Router logits are also the classic bf16 precision casualty -
computing them in fp32 and applying z-loss on the router logits (penalising log-sum-exp
magnitude) is standard and cheap.

Concrete mechanism: the z-loss keeps router logits from drifting to large magnitudes where the
softmax saturates; saturation is what makes routing decisions brittle and gradients vanish for
non-selected experts. This is a numerics fix, not a modelling fix, and it is one of the few
MoE knobs with a clear mechanism-to-symptom chain.

2) Non-determinism and debuggability. Because assignment depends on the other tokens in the
batch (capacity is shared), MoE output for a given input depends on its batch-mates. Two
identical requests batched differently can produce different logits. That breaks naive
regression tests, complicates prefix caching validation, and makes bug reports hard to
reproduce. Expert-choice routing makes this worse, token-choice with generous capacity makes
it milder but never eliminates it at capacity limits.

Boundary condition: batch-dependence disappears only when no expert is at capacity. So the
system is deterministic in the low-load regime you test in and non-deterministic in the
high-load regime you ship in. Any determinism guarantee you make must be qualified by
concurrency and capacity factor.

Evidence required: run the same prompt at concurrency 1 and at target concurrency and diff the
logits/outputs; record per-layer capacity saturation. Falsifiable claim: at capacity_factor C
and target load, output divergence rate is below X%. Rollback gate: if a correctness-critical
path (tool-call JSON, structured output) diverges under batching, either raise capacity factor
until saturation is zero or route that traffic to a dense model."""

A[284] = """Two failure modes / trade-offs of MoE.

1) Fabric sensitivity of the dispatch/combine all-to-all. Expert parallelism turns each MoE
layer into two all-to-all collectives. All-to-all stresses bisection bandwidth and is far less
forgiving than all-reduce: there is no ring or tree to hide latency behind, and every rank talks
to every rank. On a multi-node RoCE cluster this exposes the whole lossless-Ethernet stack -
PFC watchdog events, ECN marking thresholds, ARP/GID misconfiguration, and NIC-to-GPU affinity.
Without GPUDirect RDMA the payload additionally bounces through host memory, adding a PCIe
round trip per hop.

Concrete mechanism: NCCL will pick a transport per pair of ranks. If GDR is unavailable or
disabled for a NIC/GPU pair (wrong PCIe topology, missing nvidia-peermem, ACS enabled), it
silently falls back to a staged host copy. Throughput drops several-fold with no error. Verify
explicitly with NCCL_DEBUG=INFO plus NCCL_DEBUG_SUBSYS=NET and confirm GDR is in use per rank;
do not infer it from documentation.

2) Operational blast radius of a single slow expert rank. Because all-to-all is a barrier, the
slowest rank sets step time for the entire job. A single GPU with a lower clock (thermal
throttling), a degraded NIC link negotiated at half width, or an unbalanced expert assignment
stalls all N ranks. Dense TP jobs degrade the same way but MoE hits the barrier twice per layer
instead of once, so the amplification is larger.

Boundary condition: within one NVLink domain neither issue is usually visible; both appear when
the expert group crosses the node boundary, and get worse as tokens_per_rank shrinks (long
context with high TP, or decode).

Evidence required: nccl-tests alltoall bandwidth at your message size across the actual
placement; per-rank step-time histogram; link width/speed and PFC/ECN counters. Rollback gate:
if measured all-to-all bandwidth is below ~60-70% of the single-node figure, do not scale out -
fix the fabric first. All thresholds here must be measured on this cluster, not assumed."""

A[285] = """Two failure modes / trade-offs of MoE.

1) The dense-equivalent comparison is easy to get wrong, and that mis-scoping is itself the
failure. MoE papers report iso-active-FLOP or iso-training-compute wins. Serving teams then
compare against a dense model of the same total parameter count and conclude MoE is a huge win,
or against one of the same active parameter count and conclude it is free. Neither is the
decision-relevant comparison. The right one is: at fixed GPU count, fixed HBM, and fixed SLO,
which model delivers more requests/s at acceptable quality?

Concrete mechanism: HBM is partitioned between weights, KV cache, and activation workspace.
MoE takes the weight slice up, which shrinks the KV slice, which lowers the maximum concurrent
sequences, which lowers throughput even though per-token FLOPs fell. The chain runs through
KV capacity, not through FLOPs, and that is why FLOP-based reasoning misleads here.

2) Quantization and kernel maturity lag. Expert GEMMs are many small/medium GEMMs rather than
one large one; grouped-GEMM kernels, fused dispatch, and quantized expert paths are less mature
and less uniformly supported than their dense equivalents across serving stacks. Features you
take for granted on dense models (a particular quant format, speculative decoding, prefix
caching interactions, disaggregated prefill/decode handoff of the kind Mooncake or NVIDIA
Dynamo orchestrate) may be unsupported or unoptimised for MoE in the version you are running.
Verify feature support against the exact release you deploy rather than the project README.

Boundary condition: these are engineering-maturity issues, so they decay with time and are
version-specific. That means the correct answer to "should we use MoE" has a shelf life and
must be re-measured per stack version.

Evidence required: iso-hardware, iso-SLO throughput benchmark of both candidates; HBM breakdown
by weights/KV/activations; explicit feature-support check in the deployed version. Rollback
gate: fail to beat the dense baseline on requests/s at the p95 SLO, and MoE does not ship."""

A[286] = """How MoE interacts with latency, throughput, and memory.

Memory. Total parameters scale with the number of experts N; active parameters scale with k.
HBM must hold all experts that a rank owns regardless of how few tokens select them. In training
add optimizer state on top (with Adam roughly 3-4x the weight bytes depending on master-weight
and moment precision). In serving, weight bytes come directly out of the budget otherwise
available to the KV cache, and KV capacity is what sets maximum concurrency. So MoE trades
FLOPs for memory, and memory is usually the binding constraint on serving nodes.

Throughput. In the compute-bound regime - prefill, or decode at high concurrency - MoE wins,
because per-token FLOPs are k/N of the equivalent-capacity dense model. This is the regime the
design targets.

Latency. In the memory-bandwidth-bound regime - low-concurrency decode - MoE is worse per token.
Each token must read k expert weight matrices from HBM, and with poor batching those reads are
not amortised across a batch, so time-per-output-token is dominated by weight traffic that grew
rather than by FLOPs that shrank. Add all-to-all latency twice per layer, which is a fixed cost
largely independent of batch size and therefore proportionally worst at small batch.

Concrete mechanism: arithmetic intensity. Per expert GEMM, intensity is roughly
(tokens_routed_to_that_expert) FLOPs per weight byte. Total tokens split k/N ways across experts,
so each expert GEMM sees a small fraction of the batch and its intensity is correspondingly
lower than the dense FFN's. Below the accelerator's ridge point the GEMM is bandwidth-bound and
extra FLOP savings buy nothing.

Boundary condition: the crossover concurrency where MoE overtakes dense on tokens/s. Above it,
expert GEMMs are compute-bound and MoE wins; below it, dense wins. Its location depends on HBM
bandwidth, k, expert size, and all-to-all latency - it must be measured, not assumed.

Evidence and rollback gate: sweep concurrency and record TTFT, TPOT, tokens/s, plus a profile
split into attention / expert GEMM / all-to-all. If the deployment's actual concurrency
distribution sits below the measured crossover, MoE is the wrong choice for that tier."""

A[287] = """How MoE interacts with latency, throughput, and memory.

Assume a transformer decoder, top-k token-choice routing, expert parallelism, and separable
prefill and decode phases.

Prefill / throughput. Prefill batches thousands of tokens per step, so every expert receives
enough tokens for its GEMM to be compute-bound. Here MoE delivers close to its theoretical
advantage: FLOPs per token are k/N of the capacity-matched dense model. All-to-all volume is
proportional to tokens x k x d_model, so it grows with the batch, but so does the compute it
overlaps with, and the ratio stays roughly constant. Prefill is where MoE looks best.

Decode / latency. Decode processes one token per sequence per step. Per-layer all-to-all
latency is now a fixed serial cost added to every token's critical path, and expert weight reads
are amortised over far fewer tokens. TPOT therefore degrades relative to dense unless
concurrency is high enough to refill the expert GEMMs.

Concrete mechanism that exploits this asymmetry: prefill/decode disaggregation. Run prefill and
decode on separate pools with different parallelism, and ship the KV cache between them - the
pattern behind Mooncake's KV-cache-centric architecture and behind NVIDIA Dynamo's disaggregated
serving. For MoE this is more valuable than for dense models precisely because the two phases
sit on opposite sides of the arithmetic-intensity ridge, so they want different expert-parallel
degrees and different batching policies. The transfer itself then needs RDMA (GDR, or GDS if it
is staged through NVMe) or the handoff becomes the new bottleneck.

Boundary condition: disaggregation only pays when KV transfer time is small relative to the
prefill time it lets you overlap. For short prompts the transfer dominates and the aggregated
deployment wins. There is a prompt-length threshold below which disaggregation is a net loss.

Evidence required: measured KV transfer bytes and achieved RDMA bandwidth for the transfer path;
TTFT/TPOT before and after; per-phase GPU utilisation. Falsifiable claim: disaggregation
improves throughput at the deployment's median prompt length. Rollback gate: p95 TTFT regression
beyond the SLO, or KV transfer occupying more than ~15% of TTFT, reverts to aggregated serving."""

A[288] = """How MoE interacts with latency, throughput, and memory.

Start from where the bytes and the time go.

Memory. Per rank: weights = (experts owned) x expert_size + shared (attention, embeddings);
plus activations, plus - in serving - the KV cache. MoE inflates the first term while leaving
the KV term untouched, so it compresses the concurrency budget. Note that the KV cache is a
function of layers, heads, head_dim, and sequence length, and is unaffected by how many experts
exist; this is why MoE improves compute efficiency without improving long-context serving
economics at all.

Throughput. Active FLOPs per token drop by roughly k/N versus a dense model with the same total
FFN capacity. Realised throughput gain is always less, because (a) the router and the
permutation/unpermutation of tokens cost real time, (b) dispatch and combine all-to-all are
serial with the expert GEMM, and (c) load imbalance means the slowest expert sets the step time
while others idle.

Latency. Two extra collectives per MoE layer on the critical path. With 30-60 layers that is
60-120 additional synchronisation points per forward pass, each with its own launch and network
latency floor. In decode these floors do not amortise.

Concrete mechanism worth naming: token permutation. Before the expert GEMM, tokens must be
sorted by destination expert into contiguous buffers; after, they must be scattered back and
weighted by router probabilities. These gather/scatter kernels are pure HBM traffic with zero
FLOPs and are a measurable fraction of MoE layer time - fusing them into the dispatch is one of
the highest-leverage MoE kernel optimisations, and whether your stack does it is a
version-specific fact to verify by profiling, not to assume.

Boundary condition: the FLOP saving is realised only when expert GEMMs are compute-bound. Once
tokens_per_expert_per_rank falls below the level that saturates the tensor cores - long context
with high parallel degree, or low-concurrency decode - the saving disappears and only the
overheads remain.

Evidence required: kernel-level profile attributing time to router, permute, expert GEMM,
all-to-all, unpermute; tokens-per-expert histogram. Rollback gate: measured end-to-end speedup
below ~1.3x versus the dense baseline at target load does not justify the operational
complexity."""

A[289] = """How MoE interacts with latency, throughput, and memory.

Framed as the three budgets an infrastructure engineer actually manages.

HBM budget. MoE raises resident weight bytes by roughly N/k relative to an active-FLOP-matched
dense model, so on fixed hardware you either add GPUs, shard experts more aggressively, or
quantize. Expert parallelism spreads weights but sets a hard floor on world size: the model can
no longer be canaried on one GPU, which changes your whole rollout and rollback procedure.

Network budget. Dispatch + combine all-to-all per layer, roughly
tokens_per_rank x k x d_model x dtype_bytes each way. Inside an NVLink domain this is cheap.
Across nodes it needs RDMA with GPUDirect actually enabled, and it is bisection-bandwidth-bound,
so it degrades non-linearly with oversubscribed spine links or a mis-sized rail-optimised
topology.

Time budget. Throughput improves in compute-bound phases; per-token latency at low concurrency
usually worsens. Net effect on a real workload depends entirely on the concurrency distribution.

Concrete mechanism for the memory/latency trade: expert offload. Keeping cold experts in host
memory or NVMe and staging them in on demand (GDS-style direct storage-to-GPU paths avoid the
CPU bounce buffer) lowers HBM pressure. The cost is a load on the critical path whose latency is
governed by PCIe or storage bandwidth, orders of magnitude below HBM. It is viable only if
routing is predictable enough to prefetch, i.e. if expert reuse across consecutive tokens is
high.

Boundary condition: offload works when (expert reuse rate) x (prefetch lead time) covers the
transfer latency. For batch=1 chat decode, routing changes per token and prefetch fails, so
offload turns into a stall per layer. It is a batch/offline technique, not an interactive-serving
technique, unless you can demonstrate reuse.

Evidence required: expert-reuse statistics across consecutive decode steps; measured PCIe/NVMe
bandwidth on the actual path; TPOT with and without offload. Falsifiable claim: expert reuse
exceeds the level needed to hide transfer latency at the deployment's batch size. Rollback gate:
any TPOT regression past the SLO, or measured prefetch hit rate below ~80%, disables offload."""

recs = {
    280: ("rewrite", 3, 2, 3, 0.72),
    281: ("rewrite", 3, 2, 3, 0.74),
    282: ("rewrite", 3, 2, 3, 0.73),
    283: ("rewrite", 3, 2, 3, 0.71),
    284: ("rewrite", 3, 2, 3, 0.72),
    285: ("rewrite", 3, 2, 3, 0.70),
    286: ("rewrite", 3, 2, 3, 0.74),
    287: ("rewrite", 3, 2, 3, 0.72),
    288: ("rewrite", 3, 2, 3, 0.73),
    289: ("rewrite", 3, 2, 3, 0.71),
}

RISKS = [
    "Source answer is a single generic sentence; it does not supply the concrete mechanism or boundary condition the instruction explicitly requires.",
    "Generic MoE statements risk being read as platform-specific guarantees; all thresholds (crossover concurrency, drop rate, bandwidth) are cluster-dependent and must be measured.",
    "No numeric claim in the rewrite is a measured fact on any specific hardware; treating them as such would be an operational safety hazard.",
]
EVID = [
    "Kernel/step-level profile splitting time into router, permute, expert GEMM, all-to-all, unpermute.",
    "Per-layer per-expert token counts and token drop rate at the shipped capacity factor.",
    "HBM residency breakdown (weights vs KV cache vs activation workspace) on the target accelerator.",
    "Concurrency sweep of TTFT/TPOT/tokens-per-second versus a dense baseline on identical hardware.",
    "NCCL transport verification (NCCL_DEBUG=INFO, SUBSYS=NET) confirming GPUDirect RDMA is actually in use for cross-node ranks.",
]

lines = open(CORPUS).read().splitlines()
out = []
for i in range(START, START + N):
    d = json.loads(lines[i])
    msgs = {m['role']: m['content'] for m in d['messages']}
    dec, tc, ic, os_, conf = recs[i]
    out.append({
        "source_id": d['id'],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": dec,
        "source_user": msgs['user'],
        "source_assistant": msgs['assistant'],
        "corrected_answer": A[i],
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": os_,
        },
        "risks": RISKS,
        "evidence_required": EVID,
        "confidence": conf,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(out))
