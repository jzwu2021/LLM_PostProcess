import json, os

SRC = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0021.jsonl"
START, N = 200, 10

rows = [json.loads(l) for l in open(SRC)][START:START + N]

A = {}

A["corpus-00225"] = (
"Measurement plan for deciding whether tensor parallelism (TP) helps a serving workload.\n"
"Step 0, fix the question. TP helps if and only if it either (a) makes a configuration runnable that otherwise OOMs, or (b) reduces time-per-output-token (TPOT) or raises throughput at a fixed SLO. Decide which of the two you are testing before measuring, because they have opposite optimal TP degrees.\n"
"Step 1, characterise the hardware. Run nvidia-smi topo -m and record whether the candidate TP group is NVLink/NVSwitch or PCIe. Run nccl-tests all_reduce_perf -b 8 -e 128M -f 2 -g N and record both small-message latency (this governs decode) and large-message bus bandwidth (this governs prefill). Without this the rest of the plan is uninterpretable.\n"
"Step 2, sweep TP in {1,2,4,8}, holding model, dtype, max_model_len, and scheduler settings constant. At each degree record: per-rank peak HBM, KV cache blocks, TTFT p50/p99, TPOT p50/p99, and output tokens/sec, at a fixed input/output length distribution and at least three concurrency levels (1, 8, 64). Concurrency 1 isolates latency; high concurrency isolates throughput.\n"
"Step 3, establish the noise band first. Repeat the TP=1 run five times and compute the spread. Any TP effect smaller than that band is not a result.\n"
"Concrete mechanism being tested: in a Megatron-style shard the MLP up-projection is column-parallel and the down-projection is row-parallel, so each transformer block emits per-rank partial sums that require one ncclAllReduce; attention adds a second. Per-token GEMM time therefore falls roughly as 1/N while the collective term stays flat or grows, which is exactly the tradeoff the sweep is resolving.\n"
"Boundary condition: TP must stay inside one high-bandwidth domain, and the TP degree must divide num_key_value_heads. On 8x A30 24GB over PCIe Gen4 (no NVLink) the collective term is large, so TP=2 or TP=4 typically wins and TP=8 is often a regression; crossing a node boundary with TP is almost always wrong.\n"
"Falsifiable hypothesis: TPOT(N) = C/N + L(N). If the measured curve is flat or U-shaped rather than decreasing, the deployment is collective-bound and TP is not buying compute scaling.\n"
"Evidence to collect: topo matrix, nccl-tests latency and bandwidth curves, profiler split of GEMM versus ncclAllReduce per decode step, per-rank memory and KV block counts, and separate TTFT/TPOT so prefill and decode are not conflated.\n"
"Rollback gate: ship the smallest TP degree that fits memory; promote a larger degree only if it improves the primary metric by more than the measured noise band, and roll back immediately on any NCCL watchdog timeout, any rank OOM, or p99 TTFT regression above 10 percent."
)

A["corpus-00227"] = (
"Assumptions that must be stated before any performance claim about tensor parallelism is meaningful.\n"
"1. Interconnect. State whether the TP group is NVLink/NVSwitch or PCIe, and the measured (not datasheet) peer bandwidth and small-message latency. A TP=8 speedup on NVSwitch says nothing about TP=8 on PCIe Gen4.\n"
"2. Phase. State whether the number is prefill or decode. Prefill all-reduces are large and bandwidth-bound; decode all-reduces at s=1 are tiny and latency-bound. A single 'TP speedup' number that mixes them is uninterpretable.\n"
"3. Metric and load point. State TTFT versus TPOT versus tokens/sec, the concurrency level, and the input/output length distribution. TP can improve single-stream TPOT while reducing aggregate throughput versus running N independent data-parallel replicas.\n"
"4. Model shape. State num_attention_heads, num_key_value_heads, hidden size, dtype, and whether the TP degree divides them. With GQA and 8 KV heads, TP=16 forces KV replication and silently erases the memory saving.\n"
"5. Baseline and noise band. State what TP=1 (or the incumbent degree) measured, over how many repeats, and the run-to-run spread.\n"
"6. What else changed. Quantisation, attention backend, CUDA graph capture, chunked prefill, and scheduler settings must be held constant, or the claim is confounded.\n"
"Concrete mechanism this rests on: the row-parallel output projection produces per-rank partial sums that are only correct after an all-reduce over the TP communicator, and that collective sits synchronously on the critical path of every block, twice per block per forward pass. Every assumption above changes the size or cost of that term.\n"
"Boundary condition: the assumptions only bound a claim within one node and one fabric. Once the TP group spans nodes, the collective traverses RoCE/IB and, without correctly enabled GPUDirect RDMA, adds host bounce-buffer copies that invalidate any intra-node measurement.\n"
"Falsifiable check: restate the claim with the assumptions filled in and re-run on a second identical node. If the number does not reproduce within the stated noise band, the assumption list is incomplete.\n"
"Evidence to collect: nvidia-smi topo -m, nccl-tests latency and bandwidth, model config head counts, profiler GEMM-versus-NCCL split, five-repeat baseline spread, and the full server launch command line.\n"
"Rollback gate: treat any performance claim missing interconnect, phase, or baseline as unproven and do not use it to change a production TP degree; revert configuration changes justified by such claims."
)

A["corpus-00228"] = (
"Before claiming tensor parallelism improves performance, the following must be explicit, because each one can flip the sign of the result.\n"
"Assumption A, memory feasibility versus speed. TP is mandatory when weights plus activations plus KV cache exceed one device's HBM. In that regime 'TP is faster' is meaningless because the baseline does not run. State which regime you are in.\n"
"Assumption B, communication cost model. State the per-block all-reduce volume, roughly 2 * b * s * h * sizeof(dtype) * 2(N-1)/N bytes per forward pass, and the measured bandwidth and latency at that message size. Without it, no prediction is testable.\n"
"Assumption C, divisibility. num_key_value_heads must be divisible by the TP degree, otherwise the framework pads or replicates KV, wasting FLOPs or memory.\n"
"Assumption D, comparison target. TP=N on one node must be compared against the honest alternative, which is usually N data-parallel replicas of the same model. For throughput at high concurrency, DP replicas usually win; for single-stream latency, TP wins.\n"
"Assumption E, steady state. Numbers must come from a warmed server after CUDA graph capture and allocator stabilisation, not from the first requests.\n"
"Concrete mechanism: attention is sharded by heads, so each rank owns num_heads/N heads and their KV blocks, and the output projection is row-parallel with one all-reduce. This is why per-rank KV bytes fall as 1/TP only up to TP = num_key_value_heads and then stop falling; a claim that ignores this will over-predict the memory saving.\n"
"Boundary condition: on 8x A30 24GB connected over PCIe with no NVLink, the flat latency term L(N) is large enough that decode TPOT often stops improving past TP=2 to TP=4, so any claim of near-linear scaling to TP=8 on that hardware should be treated as false until shown otherwise.\n"
"Falsifiable check: per-rank KV cache bytes should scale as 1/min(TP, num_key_value_heads). If the reported KV block count does not follow that, KV is being replicated and the memory claim is wrong.\n"
"Evidence to collect: model config, per-rank allocator snapshot or server KV block count, nccl-tests at the exact implied message sizes, TTFT/TPOT at TP in {1,2,4,8}, and a DP-replica comparison at equal GPU count.\n"
"Rollback gate: do not promote a TP change into serving unless the win exceeds the five-repeat noise band and a soak run shows zero NCCL collective timeouts and per-rank step-time spread under 10 percent."
)

A["corpus-00229"] = (
"Assumptions required before a tensor-parallelism performance claim can be audited, framed as the things a reviewer should refuse to accept implicitly.\n"
"1. That the collective is actually on the critical path as measured, not assumed. State the profiler-derived fraction of step time spent in ncclAllReduce.\n"
"2. That all ranks are healthy and unthrottled. State clocks, power cap, and throttle reasons from nvidia-smi -q. A single thermally capped GPU makes the whole TP group slow, and the resulting curve looks like a communication problem but is not.\n"
"3. That the measurement is not conflating parallelism with an unrelated change. Attention backend selection, CUDA graph capture, and chunked prefill often change together with TP degree in real deployments.\n"
"4. That the workload shape is representative. TP benefits scale with per-token GEMM work, so a claim measured at batch 1, short context does not transfer to long-context prefill-heavy traffic.\n"
"5. That the failure model is accepted. TP makes the group fate-sharing; one crashed rank hangs the others until the NCCL timeout fires. A latency win that raises correlated failure probability is a business decision, not a pure performance one.\n"
"Concrete mechanism: TP is intra-operator model parallelism. A single GEMM is decomposed across N devices, each computes a partial result, and a blocking in-layer collective reassembles the mathematically exact output, so per-step time equals the slowest rank's time plus collective time. That 'slowest rank' term is why assumption 2 is not optional.\n"
"Boundary condition: this audit holds within one node. Across nodes, every block's all-reduce crosses the network, and unless GPUDirect RDMA is verified active the copy path adds host bounce buffers; multi-node TP should be treated as unsupported for latency-sensitive serving.\n"
"Falsifiable check: if per-step time variance grows superlinearly with TP degree, suspect a straggler rank before suspecting the fabric; confirm with a per-rank step-time histogram.\n"
"Evidence to collect: per-rank step timing histogram, NCCL_DEBUG=INFO topology dump, nvidia-smi -q throttle reasons, nsys trace isolating collective duration, and the exact launch command for both arms.\n"
"Rollback gate: drain back to the last known-good TP degree on any collective timeout in soak, any per-rank spread above 10 percent, or any unexplained variance growth."
)

A["corpus-00230"] = (
"The minimum assumption set that makes a tensor-parallelism performance claim falsifiable.\n"
"State the hardware: GPU model, count, and the topology matrix, because the entire tradeoff is set by whether the TP group shares NVLink or PCIe. State the measured all-reduce latency at 1 KB to 1 MB (decode regime) and bandwidth at 16 MB to 128 MB (prefill regime).\n"
"State the model: parameter count, dtype, hidden size, num_attention_heads, num_key_value_heads, max_model_len. These determine both the memory arithmetic and the divisibility constraint.\n"
"State the workload: input and output length distribution, concurrency, and whether the reported metric is TTFT, TPOT, or aggregate throughput.\n"
"State the baseline and its variance, measured over at least five repeats.\n"
"State the alternative that was not chosen, normally N data-parallel replicas at the same GPU count, so the claim is a comparison rather than an assertion.\n"
"Concrete mechanism: column-then-row sharding of the MLP is chosen deliberately because the nonlinearity is elementwise on the sharded dimension, which collapses what would be two collectives into one per MLP. The remaining collective is the irreducible cost term, and every assumption above is an input to estimating it.\n"
"Boundary condition: the claim is valid only in the regime where the model fits at both TP degrees being compared. If the baseline OOMs, TP is a feasibility mechanism and no speedup claim should be made at all; if the model fits on one device on PCIe hardware, the single-device path frequently wins on both latency and cost.\n"
"Falsifiable hypothesis: with C the single-device per-token GEMM time and L(N) the collective latency, TPOT(N) = C/N + L(N). Fit C and L from measurements at N in {1,2,4,8}; if the fit requires L(N) larger than the nccl-tests latency at the corresponding message size, something other than the collective is dominating and the stated assumptions are wrong.\n"
"Evidence to collect: topo matrix, nccl-tests latency and bandwidth curves, model config, profiler GEMM-versus-NCCL breakdown, per-rank peak memory and KV blocks, five-repeat baseline spread, and an equal-GPU-count data-parallel comparison.\n"
"Rollback gate: keep the incumbent configuration as default; promote only on a win larger than the noise band, and revert on any rank OOM, NCCL timeout, or p99 TTFT regression over 10 percent in a warmed soak run."
)

A["corpus-00231"] = (
"How tensor parallelism differs between training and inference.\n"
"Shared mechanism: in both cases a layer's weight tensors are sharded across N ranks, column-parallel then row-parallel, and the row-parallel output requires an all-reduce over the TP group to turn per-rank partial sums into the exact result.\n"
"Training specifics. The backward pass adds a second collective per sharded pair, so a transformer block costs roughly 4 collectives per step rather than 2. TP must also coexist with the optimizer sharding strategy: with ZeRO/FSDP or Megatron distributed optimizer, TP shards the parameter itself while data parallelism shards optimizer state across the DP dimension, and the two communicator groups must be laid out so that the TP group sits inside the fastest domain. Activation memory dominates, so activation recomputation interacts strongly with the chosen TP degree. Sequence lengths are long and batches large, so the collectives are big and bandwidth-bound.\n"
"Inference specifics. There is no backward pass, so only forward collectives exist. The workload splits into two phases with opposite characteristics: prefill has s equal to the prompt length, so all-reduces are large and bandwidth-bound and look like training; decode has s=1, so all-reduces are tiny and pure latency, meaning bus bandwidth barely matters and round-trip latency dominates. KV cache, which does not exist in training, is sharded by attention heads, so TP directly raises the number of concurrent sequences that fit.\n"
"Why this matters: the optimal TP degree is generally not the same for training and serving the same model. Training tolerates larger TP because the collectives amortise over large messages; decode punishes large TP because latency is flat or rising with N.\n"
"Boundary condition: in inference, TP degree must divide num_key_value_heads or KV is replicated; in training, TP degree must divide head count and the TP group must not cross the node boundary, otherwise every micro-step pays network latency four times per block.\n"
"Falsifiable hypothesis: measure all-reduce cost as a fraction of step time in training and as a fraction of TPOT in decode at the same TP degree. If decode does not show a much higher latency-bound fraction at the same N, the decode path is not actually reaching the small-message regime, which usually means CUDA graphs are disabled or batching is inflating s.\n"
"Evidence to collect: nccl-tests at both large and small message sizes, training step-time breakdown, decode TPOT curve at TP in {1,2,4,8}, per-rank KV block counts, and nvidia-smi topo -m.\n"
"Rollback gate: tune training TP and serving TP independently; roll back a serving TP increase that does not cut p50 TPOT by at least 15 percent, and roll back a training TP increase that does not improve tokens/sec/GPU beyond the noise band."
)

A["corpus-00232"] = (
"Training versus inference tensor parallelism, organised by what changes and what stays the same.\n"
"Unchanged: the sharding algebra. Column-parallel first layer, row-parallel second layer, one all-reduce to restore correctness. Head-wise sharding for attention. Divisibility constraints on head counts. Fate-sharing of the TP group.\n"
"Changed 1, number of collectives. Training executes the collective in both forward and backward, roughly doubling the communication per block per step relative to inference forward-only execution.\n"
"Changed 2, message size regime. Training messages are O(b * s * h) with large b and s, landing in the bandwidth-bound regime where NVLink versus PCIe shows up as a throughput ratio. Inference decode has s=1 and small b, landing in the latency-bound regime where the same hardware difference shows up as a fixed per-token penalty that no amount of batching removes.\n"
"Changed 3, memory pressure source. In training the pressure is optimizer state and activations; TP reduces per-rank weights and activations but the optimizer state reduction comes from the data-parallel/ZeRO dimension, not from TP. In inference the pressure is weights plus KV cache, and TP reduces both, which is why serving TP is often chosen purely to enlarge the KV pool.\n"
"Changed 4, failure and restart semantics. A training job can checkpoint and restart on rank failure with bounded loss of work; a serving deployment cannot, so a hung all-reduce in serving is an availability incident. This asymmetry should push serving toward smaller TP groups than training for the same model.\n"
"Concrete mechanism: at decode time each rank runs attention over its own num_heads/N heads and its own KV blocks, then the row-parallel output projection all-reduces a [b, 1, h] tensor. That tensor may be only a few tens of kilobytes, so the cost is essentially the interconnect round trip.\n"
"Boundary condition: on PCIe-connected A30s the decode round trip is large enough that TP beyond 2 to 4 usually stops helping, while the same hardware may still show useful training scaling at TP=4 because messages are large. Do not carry a training TP degree into serving without re-measuring.\n"
"Falsifiable check: TPOT should fall roughly as C/N + L(N). If measured L(N) exceeds nccl-tests latency at the corresponding size by a wide margin, suspect CPU launch overhead or missing CUDA graphs rather than the fabric.\n"
"Evidence to collect: separate nccl-tests runs in both size regimes, training step breakdown, decode profiler trace, per-rank KV blocks, topo matrix.\n"
"Rollback gate: serving TP changes revert on any NCCL watchdog timeout or p99 TPOT regression; training TP changes revert if tokens/sec/GPU or convergence behaviour degrades beyond the measured noise band."
)

A["corpus-00233"] = (
"Tensor parallelism in training and inference, framed by the cost equation in each setting.\n"
"Training. Per optimizer step, each transformer block pays approximately 4 all-reduces (2 forward, 2 backward) of volume O(b * s * h * sizeof(dtype)). Because b*s is large, these are bandwidth-bound and their cost is predictable from nccl-tests bus bandwidth. TP competes with pipeline parallelism and data parallelism for the same GPUs, and the standard layout puts TP innermost, inside the fastest domain, then pipeline across nodes, then data parallel outermost. TP also reduces per-rank activation memory, which lets you use a larger micro-batch or drop some recomputation, and that second-order effect is often larger than the raw FLOP division.\n"
"Inference prefill. Same shape as training forward: large messages, bandwidth-bound, TP scales reasonably.\n"
"Inference decode. s=1, so all-reduce volume collapses to O(b * h * sizeof(dtype)), often tens of kilobytes. Cost is dominated by fixed launch and round-trip latency, which is flat or increasing in N. This is the regime where TP stops paying.\n"
"Concrete mechanism explaining the asymmetry: the collective count per block is fixed by the sharding pattern, not by the token count, so as tokens per step drop from thousands (training) to one (decode) the per-token communication overhead rises by orders of magnitude while per-token compute falls.\n"
"Why it matters: the same model can rationally run at TP=4 for training and TP=1 or TP=2 for serving on identical hardware, and a team that reuses the training parallelism plan for serving will ship an avoidable latency regression.\n"
"Boundary condition: this reasoning assumes the model fits at the smaller TP degree. If serving requires TP purely to fit weights plus KV cache in 24 GB per A30, then TP degree is set by memory and the latency discussion only chooses among feasible degrees.\n"
"Falsifiable hypothesis: the ratio (collective time / step time) measured in training should be materially smaller than the ratio (collective time / TPOT) measured in decode at the same TP degree. If it is not, the decode path is not in the small-message regime and should be investigated before tuning TP.\n"
"Evidence to collect: profiler traces for both training step and decode step, nccl-tests at both message-size regimes, per-rank memory at each candidate degree, and a memory-feasibility table for weights plus KV at target max_model_len and concurrency.\n"
"Rollback gate: choose the smallest TP degree that satisfies the memory table for serving; roll back any increase that fails to cut p50 TPOT by 15 percent, and hold training TP unchanged unless tokens/sec/GPU improves beyond the noise band."
)

A["corpus-00234"] = (
"What changes for tensor parallelism when moving from training to inference, and what that implies operationally.\n"
"1. Communication doubles in training. Backward adds a mirrored collective per sharded layer pair, so training pays roughly twice the per-block collective count of inference forward.\n"
"2. New state appears in inference. The KV cache does not exist in training. Under TP it is sharded by attention heads, so per-rank KV bytes scale as 1/min(TP, num_key_value_heads). This is frequently the actual reason serving uses TP, and it is a memory argument, not a speed argument.\n"
"3. The metric changes. Training optimises tokens/sec/GPU and convergence; inference optimises TTFT and TPOT against an SLO, plus concurrency. TP moves these in different directions, so a configuration that is optimal for one is routinely wrong for the other.\n"
"4. The failure budget changes. Training tolerates restart-from-checkpoint; serving does not. TP creates correlated failure across the group, so a serving deployment should prefer the smallest TP that meets memory and latency requirements, with data-parallel replicas providing redundancy.\n"
"5. Dynamic batching interacts. In serving, batch composition changes every step, so all-reduce sizes vary step to step and CUDA graph capture must handle the shapes; in training shapes are static.\n"
"Concrete mechanism: the row-parallel output projection emits per-rank partial sums that are only correct after ncclAllReduce over the TP communicator; that call is blocking, so per-step time equals the slowest rank's time plus collective time in both settings, but the ratio of those two terms differs by orders of magnitude between a 4096-token training micro-batch and a single decode token.\n"
"Boundary condition: TP must remain within one node and one fabric in both settings, and the degree must divide num_key_value_heads. On 8x A30 24GB over PCIe, decode-oriented TP typically saturates at 2 to 4 while training may still benefit at 4.\n"
"Falsifiable check: per-rank KV blocks should fall as 1/min(TP, num_key_value_heads). If they do not, KV is replicated and the serving memory rationale for raising TP is invalid.\n"
"Evidence to collect: model config head counts, server KV block counts at each TP degree, nccl-tests in both size regimes, decode and training profiler traces, per-rank step-time histogram, topo matrix.\n"
"Rollback gate: serving reverts to the previous TP degree on any NCCL watchdog timeout, rank OOM, per-rank step-time spread above 10 percent, or p99 latency regression above 10 percent in a warmed soak."
)

A["corpus-00235"] = (
"Training versus inference tensor parallelism, stated as a decision procedure rather than a description.\n"
"Step 1, compute the memory table separately for each setting. Training per-rank need is weights/N + gradients/N + optimizer state (sharded over the DP dimension, not over TP) + activations/N with recomputation policy applied. Inference per-rank need is weights/N + KV cache/min(N, num_key_value_heads) at target max_model_len and target concurrency + workspace. These give different minimum feasible TP degrees.\n"
"Step 2, among feasible degrees, pick by the setting's metric. Training: maximise tokens/sec/GPU, so favour the degree that minimises total collective time given large bandwidth-bound messages. Inference: minimise TPOT under the SLO, so favour the smallest degree, because decode collectives are latency-bound and do not amortise.\n"
"Step 3, validate independently in each setting. Do not carry the training TP degree into serving.\n"
"Concrete mechanism: TP shards a single GEMM across N devices; each holds a slice of the weight, computes a partial result, and an in-layer collective reassembles the exact output. In training that collective fires in forward and backward; in decode it fires only forward but on a tensor of shape [b, 1, h], which is too small to hide behind compute.\n"
"Why it matters: this is the most common source of a bad serving configuration in teams that train and serve the same model, because the training recipe is treated as authoritative for deployment when the two regimes have different optima.\n"
"Boundary condition: the procedure assumes a single node and a TP degree dividing num_key_value_heads. It does not apply to multi-node TP, which should be avoided for serving entirely; across nodes, prefer pipeline or data parallelism and verify GPUDirect RDMA is active before assuming any RDMA path performance.\n"
"Falsifiable hypothesis: for a model feasible at TP=1 and TP=2, serving TPOT should improve by at least 1.5x on an NVLink-class fabric and materially less on PCIe. If PCIe shows no improvement, the deployment is collective-latency-bound and TP should be set by memory feasibility alone.\n"
"Evidence to collect: the two memory tables with real allocator snapshots, nccl-tests latency and bandwidth, training step and decode profiler breakdowns, TTFT/TPOT at TP in {1,2,4,8} at concurrency 1/8/64, and five-repeat noise bands.\n"
"Rollback gate: promote a new serving TP degree only if it beats the incumbent by more than the noise band on the SLO metric with zero NCCL timeouts in soak; otherwise revert. Keep training and serving configurations versioned separately so a rollback in one does not perturb the other."
)

RISKS = [
    "the source assistant text is a single generic sentence that names no concrete sharding mechanism (column-parallel, row-parallel, head-wise KV sharding), so it fails the prompt's explicit demand for one concrete mechanism",
    "no boundary condition is supplied even though every prompt in this block requires one; 'depends on memory, topology, batch, and communication cost' is a list of variables, not a stated crossover or constraint",
    "the identical answer is reused across measurement-plan, assumption-listing, and training-versus-inference framings, so at least two of the three question types are simply not answered",
    "no falsifiable prediction, no evidence list, and no rollback gate, which makes the answer operationally unsafe as guidance for changing a serving TP degree",
    "omits the divisibility constraint on num_key_value_heads, so a reader could raise TP past the KV head count and silently lose the KV memory saving they were targeting",
]

EVID = [
    "nvidia-smi topo -m to establish whether the candidate TP group is NVLink/NVSwitch or PCIe connected",
    "nccl-tests all_reduce_perf latency at 1 KB-1 MB (decode regime) and bus bandwidth at 16-128 MB (prefill/training regime)",
    "profiler (nsys or torch profiler) split of GEMM time versus ncclAllReduce time per decode step and per training step",
    "per-rank peak HBM and server KV block counts at TP in {1,2,4,8}, checked against the 1/min(TP, num_key_value_heads) prediction",
    "TTFT and TPOT reported separately at concurrency 1, 8, and 64, with a five-repeat baseline noise band",
    "model config num_attention_heads and num_key_value_heads to verify divisibility at each candidate TP degree",
]

recs = []
for r in rows:
    sid = r["id"]
    msgs = {m["role"]: m["content"] for m in r["messages"]}
    recs.append({
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": msgs["user"],
        "source_assistant": msgs["assistant"],
        "corrected_answer": A[sid],
        "quality_dimensions": {
            "technical_correctness": 4,
            "instruction_coverage": 2,
            "operational_safety": 3,
        },
        "risks": RISKS,
        "evidence_required": EVID,
        "confidence": 0.74,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for rec in recs:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs), recs[0]["source_id"], recs[-1]["source_id"])
