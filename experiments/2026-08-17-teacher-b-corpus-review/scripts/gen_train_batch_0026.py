import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0026.jsonl")
START, END = 250, 260  # 0-indexed slice

COMMON_RISK = "Source answer is a one-line slogan: no assumption list, no units, no boundary condition, no falsifiable claim, so it can be recited without predicting any real deployment outcome."

ANS = {}

ANS["corpus-00280"] = """Assumptions that MUST be stated before any pipeline-parallelism (PP) performance claim:

1. Model/shape: decoder-only transformer, layer count L, hidden size H, sequence length S, and whether the claim is about training or decode. PP throughput math differs by an order of magnitude between the two.
2. Topology: PP degree P, placement of stage boundaries (intra-node NVLink vs inter-node RDMA/RoCE), and the measured point-to-point bandwidth/latency on the boundary link. "200 Gb/s NIC" is a nameplate number, not a measured one; state the ib_write_bw or nccl-tests p2p result you actually got.
3. Schedule: GPipe vs 1F1B vs interleaved (virtual pipeline) 1F1B, and the number of microbatches M per global batch. Without M and the schedule, the bubble term is undefined.
4. Composition: TP degree, DP degree, ZeRO stage, activation recomputation on/off. PP numbers are not comparable across different recomputation settings because recomputation trades ~30% extra FLOPs for activation memory.
5. Measurement protocol: warmup steps discarded, steady-state window, whether the timer includes optimizer step and gradient all-reduce, and whether numbers are per-GPU or aggregate.

Concrete mechanism: PP splits the layer stack into P sequential stages; only stage-boundary activations of shape [micro_batch, S, H] cross the boundary link, once forward and once backward. So PP inter-stage traffic is O(P) point-to-point sends per microbatch, versus TP's per-layer all-reduce. With M microbatches, the GPipe bubble fraction is (P-1)/(M+P-1); 1F1B keeps the same bubble fraction but bounds peak activation memory to about P in-flight microbatches on stage 0 instead of M.

Boundary condition: the claim inverts when M < P. At M = P the bubble is ~50% and PP is worse than the same GPU count spent on TP or DP. It also inverts when per-hop activation transfer time exceeds per-stage compute time (typical if the boundary crosses a TCP/Ethernet link rather than RDMA); then adding stages monotonically reduces throughput.

Falsifiable claim: with fixed global batch, measured step time should follow t(M) ≈ t_compute * (1 + (P-1)/M) plus a constant per-hop term. Sweep M ∈ {P, 2P, 4P, 8P} and fit; if the residual does not shrink like 1/M, the bottleneck is the link or a load-imbalanced stage split, not the bubble.

Evidence required: per-stage forward/backward time histogram (to prove stages are balanced within ~5%), nccl-tests or ib_write_bw on the boundary link, torch profiler or Megatron timers separating compute from send/recv wait, and nvidia-smi / DCGM utilization per stage.

Rollback gate: if end-to-end tokens/s at the new P is not >= 95% of the previous configuration at equal GPU count, or if any stage shows >15% send/recv wait, revert to the prior parallelism plan before merging."""

ANS["corpus-00281"] = """Assumptions: decoder-only transformer, PP degree P, homogeneous GPUs, stage boundaries on NVLink intra-node or RDMA/RoCE inter-node, KV cache resident on the stage that owns those layers.

Concrete mechanism: in both training and inference PP moves only stage-boundary activations of shape [micro_batch, S, H]. What changes is the supply of independent work items that can fill the pipe. In training you control M microbatches per global batch, so you can choose M >> P and the GPipe bubble fraction (P-1)/(M+P-1) becomes small; 1F1B further bounds peak activation memory to ~P in-flight microbatches. In inference there is no gradient pass and no microbatch knob: the pipe is filled by concurrent requests, and during decode each step carries only [batch, 1, H] per stage, so the per-hop payload is tiny while the per-hop latency is fixed.

Training side specifics: the backward pass doubles the number of stage crossings and requires stashed activations, so activation memory, not weight memory, is usually the binding constraint. Weight+optimizer memory drops ~1/P, which is the main reason PP is used at all.

Inference side specifics: PP does not reduce single-request latency; it adds P-1 hop latencies to both prefill and every decode step. It does cut per-GPU weight memory ~1/P, which is how you fit a model that does not fit under TP alone within one NVLink domain. Continuous batching partially hides the bubble because different requests occupy different stages, but only if in-flight concurrency >= P.

Boundary condition: at low concurrency (concurrency < P) inference PP degrades to serial execution with idle stages, so inter-token latency rises roughly linearly in P while throughput does not improve. In training the mirror boundary is M < P. A second boundary: if the boundary link is TCP/Ethernet rather than RDMA, the fixed per-hop cost (tens to hundreds of microseconds) becomes comparable to a decode step, and decode PP is unusable regardless of concurrency.

Falsifiable claim: in serving, hold GPU count fixed and sweep concurrency; PP throughput should approach the TP baseline only for concurrency >= ~2P, while p50 inter-token latency should stay above the TP baseline by approximately (P-1) * per-hop latency at all concurrency levels. If measured inter-token latency does not scale with P, the hop cost is being hidden by another bottleneck and your P attribution is wrong.

Evidence required: concurrency sweep with p50/p99 TTFT and inter-token latency, per-stage busy/idle fraction, boundary-link RTT and bandwidth from ib_write_lat/ib_write_bw, and memory-per-GPU before/after.

Rollback gate: revert to the previous parallelism layout if p99 inter-token latency regresses more than 10% against the SLO, or if throughput at target concurrency is below the TP baseline."""

ANS["corpus-00282"] = """Assumptions: same weights served and trained, PP degree P, 1F1B schedule in training, continuous batching in serving, stage boundaries measured (not assumed) on their actual link.

Concrete mechanism: the invariant across both regimes is that PP transfers only stage-boundary hidden states point-to-point, so its wire volume is far below TP's per-layer all-reduce. The variable is what fills the pipeline. Training fills it with M microbatches you choose; inference fills it with concurrent requests you do not control. That single difference drives every other divergence below.

Divergence 1 - the bubble knob. Training: bubble fraction (P-1)/(M+P-1) is a tunable because M is a hyperparameter; raising M costs only activation stash memory. Inference: there is no M, only arrival rate; under low load the pipe is structurally empty and stages idle.

Divergence 2 - memory pressure. Training peak memory is dominated by stashed activations for up to ~P in-flight microbatches on the first stage, plus optimizer state (~12-16 bytes/param for fp32 Adam moments and master weights, or less under ZeRO). Inference has no optimizer state and no activation stash; instead the KV cache is partitioned by layer, so each stage holds KV for its own layers only, and cache capacity per stage becomes the admission-control limit.

Divergence 3 - payload size. Training microbatch activations are [mb, S, H] and can be megabytes; decode-step activations are [batch, 1, H] and are typically kilobytes. So training PP is bandwidth-sensitive while decode PP is latency-sensitive. The right measurement for training is ib_write_bw; the right measurement for decode is ib_write_lat.

Divergence 4 - failure mode. A slow stage in training shows up as a uniform step-time increase because the schedule is synchronous. In serving a slow stage shows up as head-of-line blocking and a p99 tail, while p50 may look fine.

Boundary condition: PP stops helping when the pipe cannot be filled - M < P in training, in-flight concurrency < P in serving - or when per-hop transfer time exceeds per-stage compute time. Under either condition, increasing P monotonically worsens the metric you care about.

Falsifiable claim: the ratio of decode inter-token latency at PP=P to PP=1 should be approximately 1 + (P-1)*hop_latency/step_compute_time, and this should hold independent of batch size. Measure it; if the ratio grows with batch size, the boundary link is bandwidth-bound, not latency-bound, and the model of the system is wrong.

Evidence required: per-stage timers, ib_write_lat and ib_write_bw on the exact boundary path, KV-cache occupancy per stage, and a load sweep separating p50 from p99.

Rollback gate: revert if p99 TTFT breaches SLO at target QPS, or if training step time is not within 5% of the (P-1)/(M+P-1) prediction after stage rebalancing."""

ANS["corpus-00283"] = """Assumptions: decoder-only transformer, PP degree P, stage split by layer count, boundary links characterized by measurement, training uses 1F1B, serving uses continuous batching.

Concrete mechanism: PP is layer-sharding along depth; only the activation at each stage boundary crosses the wire. In training each microbatch crosses each boundary twice (forward activation, backward gradient w.r.t. that activation) and the first stage must stash activations for in-flight microbatches. In inference there is no backward pass, so each boundary is crossed once per prefill chunk and once per decode step, and the stashed-activation cost disappears entirely; what replaces it is the KV cache, which is naturally partitioned because each stage owns only its own layers' K/V tensors.

Practical consequences of that asymmetry:
- Training benefits from PP mainly through memory: weights + gradients + optimizer state divide by ~P. Throughput benefit only materializes when M >> P.
- Inference benefits from PP almost exclusively through weight memory and through crossing beyond one NVLink domain when TP all-reduce becomes the bottleneck. It never improves single-stream latency.
- Training is throughput-only, so a 5-10% bubble is acceptable. Serving has latency SLOs, so the same 5-10% may be unacceptable at p99.
- Recomputation is a training-only lever; in inference the corresponding lever is chunked prefill and KV paging.

Boundary condition: the regimes invert at low occupancy. Training with M < P wastes >50% of the pipe; serving with concurrency < P does the same, and additionally pays (P-1) hop latencies on every single decoded token. A second, harder boundary: if the boundary crosses a non-RDMA link, decode PP fails outright because the per-hop cost is a fixed tens-to-hundreds of microseconds against a decode step of a few milliseconds or less.

Falsifiable claim: disabling PP in favor of TP at equal GPU count should reduce decode inter-token latency by roughly (P-1)*hop_latency while leaving peak throughput at high concurrency within ~10%. If instead throughput collapses, the bottleneck was TP all-reduce bandwidth, and PP was doing real work.

Evidence required: per-stage forward time balance within 5%, measured hop latency/bandwidth (ib_write_lat, ib_write_bw, nccl-tests), token throughput and inter-token latency at multiple concurrency levels, and per-GPU memory before/after.

Rollback gate: keep the change only if training step time improves >= 5% at equal GPU count with no loss-curve divergence over 500 steps, or if serving p99 TTFT and inter-token latency both stay within SLO; otherwise revert to the previous parallelism configuration."""

ANS["corpus-00284"] = """Assumptions: fixed model weights, PP degree P, homogeneous accelerators, stage boundaries whose bandwidth and latency have been measured, and a stated schedule (1F1B for training, continuous batching for serving).

Concrete mechanism: what physically changes between training and inference is the number and size of stage-boundary crossings per unit of user-visible work. Training: 2 crossings per boundary per microbatch (forward activation [mb,S,H], backward grad of the same shape), M microbatches per step, plus an activation stash of up to ~P microbatches on the earliest stage under 1F1B. Inference: 1 crossing per boundary per prefill chunk (payload [chunk,S_chunk,H]) and 1 per decode step (payload [batch,1,H]), no stash, but a per-stage KV cache proportional to that stage's layer count times total live tokens.

Therefore:
- Training is dominated by bandwidth and memory; inference decode is dominated by fixed per-hop latency.
- Training can hide the bubble by raising M for free (subject to activation memory); serving cannot, because arrival rate is exogenous.
- Training failure of a stage split shows as uniformly slower steps; serving failure shows as a p99 tail with an acceptable p50.

Boundary condition: PP is only a win while the pipe stays full. In training that means M >= 4P as a practical rule (bubble <= 20% at M=4P for the classic formula). In serving it means sustained in-flight concurrency >= 2P. Below those points, spending the same GPUs on TP (inside an NVLink domain) or on DP replicas beats PP. The second boundary is link class: PP boundaries on TCP/Ethernet are viable for training (large payloads amortize the overhead) and generally not viable for decode.

Falsifiable claim: for training, measured step time should fit t_compute*(1 + (P-1)/M) + c within 10% across M ∈ {P,2P,4P,8P}. For serving, measured inter-token latency should be independent of batch size but linear in P. If either fit fails, the limiting resource is not the pipeline bubble - look for stage imbalance or link saturation before changing P again.

Evidence required: per-stage timing histograms, ib_write_lat / ib_write_bw or nccl-tests p2p on the exact boundary path, torch profiler traces showing send/recv wait fraction per stage, per-GPU memory high-water mark, and for serving a concurrency sweep reporting p50/p99 TTFT and inter-token latency.

Rollback gate: revert the parallelism change if any stage shows >15% send/recv wait after rebalancing, if training tokens/s per GPU regresses at all, or if serving p99 breaches the latency SLO at target QPS."""

ANS["corpus-00285"] = """Assumptions: same checkpoint used for both regimes, PP degree P, layer-balanced stage split, measured boundary-link characteristics, 1F1B in training and continuous batching in serving.

Concrete mechanism: PP shards depth, so only stage-boundary hidden states move. The regime difference is entirely about what fills the pipeline and what dominates memory.
- Training: M microbatches per optimizer step fill the pipe; bubble fraction is (P-1)/(M+P-1) and is under your control. Memory is dominated by stashed activations (bounded to ~P in-flight microbatches under 1F1B) plus optimizer state, which PP divides by ~P.
- Inference: concurrent requests fill the pipe; there is no M knob. Memory is dominated by the KV cache, which PP partitions by layer so each stage holds only its own layers' K/V. Latency picks up P-1 hop costs on prefill and on every decode step.

Second-order effects worth stating: PP in training must be co-designed with gradient accumulation and with the DP all-reduce, which happens only at step boundaries and can overlap with the pipeline tail. In serving, PP interacts with scheduling policy - a prefill on stage 0 can stall decode microbatches behind it unless prefill is chunked, which is the serving analogue of a load-imbalanced stage.

Boundary condition: the benefit disappears when occupancy falls below P (M < P in training, concurrency < P in serving), and it inverts when per-hop transfer time exceeds per-stage compute time. Concretely, a decode step whose per-stage compute is ~1 ms cannot tolerate a 300 microsecond TCP hop times P-1 hops; on RDMA/RoCE with a few microseconds per hop the same configuration is fine. This is why the same PP degree can be correct for the training job and wrong for the serving deployment of the identical model.

Falsifiable claim: if you deploy the trained model at the same PP degree used for training, decode inter-token latency will exceed the TP-only deployment by approximately (P-1)*measured_hop_latency, and this gap will not close as batch size grows. Measuring a gap that grows with batch size falsifies the latency-bound model and indicates the boundary link is bandwidth-saturated instead.

Evidence required: per-stage busy/idle fractions, ib_write_lat and ib_write_bw on the boundary path, concurrency sweep with p50/p99 TTFT and inter-token latency, KV-cache occupancy per stage, and training step-time versus M sweep.

Rollback gate: do not carry the training parallelism plan into serving unless the concurrency sweep shows p99 within SLO; otherwise revert to the TP-only or lower-P serving layout that last met SLO."""

ANS["corpus-00286"] = """Misleading intuition: "Pipeline parallelism with P stages gives roughly Px throughput, the same way data parallelism does."

Why it is wrong: DP replicates the whole model and processes independent batches, so its scaling is near-linear until the gradient all-reduce saturates. PP splits one model into a serial chain; a single work item still traverses all P stages, so the latency of one item does not improve at all, and aggregate throughput improves only to the extent that you can keep every stage simultaneously busy with different in-flight items. PP's headline benefit is memory - weights, gradients and optimizer state divide by ~P - not raw speed.

Concrete mechanism: with M microbatches and P stages under GPipe, the fill and drain phases leave a bubble of fraction (P-1)/(M+P-1). At M=P the pipe is idle ~50% of the time. 1F1B does not shrink that bubble; it bounds peak activation memory to about P in-flight microbatches so you can afford a larger M, which is what actually shrinks the bubble. Interleaved (virtual pipeline) 1F1B with v virtual chunks per device reduces the bubble to about (P-1)/(v*M+P-1) at the cost of v times more boundary crossings, so it trades link traffic for bubble.

Correct statement: PP scales capacity (model size you can fit and total GPUs you can use past the TP all-reduce limit), and scales throughput only sub-linearly, gated by pipeline occupancy.

Boundary condition: the intuition is least wrong when M >> P and the boundary links are fast (RDMA/RoCE or NVLink) so per-hop cost is negligible against per-stage compute. It is most wrong in decode serving, where in-flight concurrency is small, payloads are [batch,1,H], and each of the P-1 hops adds fixed latency to every token.

Falsifiable claim: hold GPU count constant and compare PP=P against DP=P on a model that fits in one GPU. DP should give near-linear tokens/s scaling; PP should give strictly less, with the gap tracking (P-1)/(M+P-1). If PP matches DP, either M is very large and the bubble has vanished, or your DP run is all-reduce bound - check the network before crediting PP.

Evidence required: side-by-side tokens/s at equal GPU count, per-stage send/recv wait fraction, M sweep, and measured boundary-link bandwidth/latency.

Rollback gate: adopt PP only if it is required to fit the model or to exceed the TP domain; if a DP or TP configuration at equal GPU count meets the memory constraint and beats PP throughput, revert."""

ANS["corpus-00287"] = """Misleading intuition: "Pipeline parallelism reduces per-request latency because the work is spread over more GPUs."

Why it is wrong: PP is a serial decomposition. A request must pass through stage 0, then 1, ... then P-1 in order. The total compute per request is unchanged (it is the same layers, just on different devices), and you additionally pay P-1 inter-stage transfers. So single-request latency strictly increases with P. Only tensor parallelism reduces the latency of a single layer's math by splitting the matmul itself; PP reduces per-GPU memory and increases aggregate throughput under sufficient concurrency.

Concrete mechanism: in decode, each generated token requires a full traversal of the chain. Per token you pay sum(per-stage compute) + (P-1) * hop_cost, where hop_cost = serialization of a [batch, 1, H] tensor plus link RTT plus any kernel-launch/sync overhead. On RDMA/RoCE a hop is a few microseconds; on TCP over Ethernet it is often hundreds of microseconds. With H=8192 and fp16, [batch=32,1,8192] is only ~0.5 MB, so hop cost is latency-dominated, not bandwidth-dominated - which is precisely why the intuition fails: more bandwidth does not fix it.

Correct statement: PP trades single-request latency for capacity. Use it when the model does not fit otherwise, or when TP has already saturated its NVLink domain and further TP would be all-reduce bound.

Boundary condition: the penalty is masked only when per-stage compute is large relative to hop cost - long prefill chunks, big batches, low P, RDMA boundaries. It is maximal in low-concurrency interactive decode with inter-node boundaries, where inter-token latency can be dominated by hops rather than by matmuls.

Falsifiable claim: run a single request (batch=1) at PP=1 and PP=P on the same GPU class. Inter-token latency at PP=P should exceed PP=1 by approximately (P-1)*measured_hop_latency, with the per-stage compute sum roughly unchanged. If PP=P is faster at batch=1, something else changed - check that PP=1 was not memory-thrashing or offloading.

Evidence required: batch=1 inter-token latency at both configurations, ib_write_lat on the exact boundary path, per-stage kernel time from the profiler, and confirmation that no offload/swap is active in the baseline.

Rollback gate: if the deployment has an interactive latency SLO and PP=P misses p99 while a TP-only layout meets it at equal GPU count, revert to TP-only; adopt PP only where the memory constraint makes TP-only infeasible."""

ANS["corpus-00288"] = """Misleading intuition: "The pipeline bubble is fixed by the schedule, so switching GPipe to 1F1B makes the pipeline faster."

Why it is wrong: GPipe and 1F1B have the same steady-state bubble fraction, approximately (P-1)/(M+P-1). What 1F1B changes is memory: by interleaving one forward and one backward, it bounds the number of stashed activation sets on the earliest stage to about P instead of M. The speedup people observe after switching is indirect - lower activation memory lets them raise M (or disable recomputation), and it is the larger M that shrinks the bubble.

Concrete mechanism: under GPipe, stage 0 must retain activations for all M microbatches until backward begins, so peak activation memory scales with M and caps M well below what the bubble formula wants. Under 1F1B, stage s holds at most (P - s) in-flight microbatches, so peak memory is O(P) and independent of M; M can then be raised until the DP all-reduce or the global batch-size budget binds. Interleaved 1F1B with v virtual chunks per device further reduces the bubble to about (P-1)/(v*M+P-1), but multiplies boundary crossings by v - a real cost on inter-node RDMA boundaries.

Correct statement: 1F1B is a memory-schedule optimization that unlocks bubble reduction; it is not itself a bubble reduction.

Boundary condition: the distinction stops mattering when M is already large enough that the bubble is negligible (say M >= 8P, bubble ~11% at M=8P by the formula) and activation memory is not binding - then GPipe and 1F1B perform the same. It matters most at large P with tight HBM, e.g. 24 GB-class GPUs where the activation stash is the first thing to OOM. Interleaving specifically stops paying off when v * boundary traffic saturates the inter-node link.

Falsifiable claim: at fixed M, GPipe and 1F1B step times should agree within a few percent; only the peak memory high-water mark should differ, and 1F1B's should be roughly M/P times lower on stage 0. If step time changes materially at fixed M, the two runs differ in something else (recomputation flag, M, or stage split) - re-check the config before crediting the schedule.

Evidence required: peak allocated memory per stage from torch.cuda.max_memory_allocated, step time at matched M and matched recomputation setting, per-stage send/recv wait fraction, and the M sweep after the switch.

Rollback gate: keep the schedule change only if peak memory on the first stage drops as predicted and the subsequently raised M yields >= 5% tokens/s improvement with an unchanged loss curve over 500 steps; otherwise revert."""

ANS["corpus-00289"] = """Misleading intuition: "Pipeline parallelism is cheap on the network because it only sends activations, so the interconnect does not matter."

Why it is wrong: the volume claim is true and the conclusion is false. PP does move far less data than tensor parallelism - point-to-point activations at stage boundaries versus per-layer all-reduce - but PP's crossings sit on the synchronous critical path of a serial chain, and there are P-1 of them per traversal. Low volume with high per-transfer latency is exactly the regime where the interconnect class (RDMA/RoCE with GPUDirect RDMA versus TCP with a host bounce) dominates, because the cost is per-message overhead, not bytes.

Concrete mechanism: a decode-step boundary payload is [batch, 1, H]; at batch=32, H=8192, fp16 that is about 0.5 MB, which even a 25 Gb/s link moves in ~170 microseconds of pure serialization - but a non-GPUDirect path adds a device-to-host copy, a host-to-device copy, and a kernel launch/sync on each side. With GDR the NIC DMAs straight out of GPU memory and the hop can be a few microseconds. Training payloads [mb, S, H] are far larger and are bandwidth-sensitive instead, so the same PP topology can be bandwidth-bound in training and latency-bound in decode.

Correct statement: PP is bandwidth-light but latency-sensitive and topology-sensitive. Stage boundaries should be placed to minimize the number of crossings that traverse the slowest link, and GDR should be verified as actually enabled, not assumed.

Boundary condition: the "network does not matter" intuition holds only when per-stage compute time is much larger than per-hop cost - large training microbatches on RDMA boundaries. It fails when P is large, when boundaries cross a slow or oversubscribed fabric, when GDR silently falls back to a host bounce (a common misconfiguration), or in interleaved 1F1B where crossings are multiplied by v.

Falsifiable claim: measured per-hop time should match ib_write_lat for the payload size when GPUDirect RDMA is active, and should exceed it by the cost of two PCIe copies when it is not. Compare NCCL_DEBUG=INFO transport lines against measured hop time; a gap of tens of microseconds indicates a host bounce.

Evidence required: NCCL_DEBUG=INFO topology/transport output confirming the GDR path, ib_write_lat and ib_write_bw at the actual payload size, nccl-tests point-to-point results, per-stage send/recv wait fraction from the profiler, and PCIe counters if a bounce is suspected.

Rollback gate: do not increase P or enable interleaving until GDR is confirmed on every stage boundary and measured hop latency is within 2x of ib_write_lat; if a change raises any stage's send/recv wait above 15%, revert it."""

# quality_dimensions rate the SOURCE assistant answer (1-5), matching prior batches.
QD = {
    "corpus-00280": (3, 1, 2),
    "corpus-00281": (3, 1, 2),
    "corpus-00282": (3, 1, 2),
    "corpus-00283": (3, 1, 2),
    "corpus-00284": (3, 1, 2),
    "corpus-00285": (3, 1, 2),
    "corpus-00286": (3, 1, 2),
    "corpus-00287": (3, 1, 2),
    "corpus-00288": (3, 1, 2),
    "corpus-00289": (3, 1, 2),
}

RISKS = {
    "corpus-00280": [COMMON_RISK, "Prompt explicitly asks for assumptions; the source answer lists none, so an operator could quote a PP throughput number with no stated schedule, M, or link class."],
    "corpus-00281": [COMMON_RISK, "Does not distinguish training from inference at all; carrying a training PP degree into serving is a concrete latency-SLO regression risk."],
    "corpus-00282": [COMMON_RISK, "Omits that inference has no microbatch knob, which is the crux of the training/inference difference."],
    "corpus-00283": [COMMON_RISK, "Omits KV-cache partitioning and the absence of an activation stash in inference, both of which change capacity planning."],
    "corpus-00284": [COMMON_RISK, "No units, no payload shapes, and no link-class discussion, so it cannot support a sizing decision."],
    "corpus-00285": [COMMON_RISK, "Silent on the failure mode where the training parallelism plan is reused for serving and breaches p99."],
    "corpus-00286": [COMMON_RISK, "Prompt asks for a misconception and a correction; the source states neither, so the training signal for the requested task shape is absent."],
    "corpus-00287": [COMMON_RISK, "Fails to correct the common belief that PP lowers per-request latency, which is the specific error this item should teach against."],
    "corpus-00288": [COMMON_RISK, "Mentions microbatching as bubble reduction without separating the schedule's memory effect from the bubble formula, which is itself the misconception."],
    "corpus-00289": [COMMON_RISK, "No mention of GDR, RDMA/RoCE, or host-bounce fallback, so it cannot flag a silent GPUDirect misconfiguration."],
}

EV = {
    "corpus-00280": ["Per-stage forward/backward timing histogram showing stage balance within 5%", "ib_write_bw / ib_write_lat or nccl-tests p2p on the actual stage-boundary path", "Step-time vs M sweep over M in {P,2P,4P,8P}", "Profiler send/recv wait fraction per stage"],
    "corpus-00281": ["Concurrency sweep with p50/p99 TTFT and inter-token latency", "Per-stage busy/idle fraction", "Measured hop latency (ib_write_lat) at decode payload size", "Per-GPU memory high-water mark before/after"],
    "corpus-00282": ["ib_write_bw for training payloads and ib_write_lat for decode payloads on the same path", "Per-stage timers under both regimes", "KV-cache occupancy per stage", "Load sweep separating p50 from p99"],
    "corpus-00283": ["Stage balance within 5% from per-stage forward times", "nccl-tests p2p and ib_write_lat on the boundary link", "Throughput and inter-token latency at multiple concurrency levels", "Per-GPU memory before/after"],
    "corpus-00284": ["Fit of measured step time to t_compute*(1+(P-1)/M)+c across an M sweep", "torch.cuda.max_memory_allocated per stage", "ib_write_lat / ib_write_bw on the boundary path", "Concurrency sweep p50/p99 for the serving side"],
    "corpus-00285": ["Training step-time vs M sweep", "Serving concurrency sweep with p99 inter-token latency", "ib_write_lat on the boundary path", "KV-cache occupancy per stage"],
    "corpus-00286": ["Side-by-side tokens/s for PP=P vs DP=P at equal GPU count", "Per-stage send/recv wait fraction", "M sweep with bubble-fraction comparison", "Measured boundary-link bandwidth and latency"],
    "corpus-00287": ["batch=1 inter-token latency at PP=1 and PP=P on identical hardware", "ib_write_lat on the boundary path", "Per-stage kernel time from the profiler", "Confirmation that no CPU offload/swap is active in the baseline"],
    "corpus-00288": ["torch.cuda.max_memory_allocated per stage for GPipe vs 1F1B at matched M", "Step time at matched M and matched recomputation setting", "Per-stage send/recv wait fraction", "Post-switch M sweep and 500-step loss curve"],
    "corpus-00289": ["NCCL_DEBUG=INFO transport lines confirming the GPUDirect RDMA path", "ib_write_lat and ib_write_bw at the actual boundary payload size", "nccl-tests point-to-point results", "Profiler send/recv wait fraction per stage"],
}

CONF = {
    "corpus-00280": 0.82, "corpus-00281": 0.83, "corpus-00282": 0.81, "corpus-00283": 0.82,
    "corpus-00284": 0.8, "corpus-00285": 0.81, "corpus-00286": 0.84, "corpus-00287": 0.85,
    "corpus-00288": 0.8, "corpus-00289": 0.82,
}

rows = []
with open(CORPUS) as f:
    lines = f.read().splitlines()
for line in lines[START:END]:
    d = json.loads(line)
    sid = d["id"]
    msgs = d["messages"]
    su = [m for m in msgs if m["role"] == "user"][0]["content"]
    sa = [m for m in msgs if m["role"] == "assistant"][0]["content"]
    tc, ic, os_ = QD[sid]
    rows.append({
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": su,
        "source_assistant": sa,
        "corrected_answer": ANS[sid],
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": os_,
        },
        "risks": RISKS[sid],
        "evidence_required": EV[sid],
        "confidence": CONF[sid],
    })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(rows))
