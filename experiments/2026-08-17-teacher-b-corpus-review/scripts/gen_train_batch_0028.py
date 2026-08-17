import json
BASE='/home/johnson/workspace/LLM_PostProcess'
OUT=BASE+'/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0028.jsonl'
src={}
for l in open(BASE+'/research/ai-infra-expert/corpus/train.jsonl'):
    r=json.loads(l); src[r['id']]=r

PP_RUNBOOK = """Runbook entry: investigating a pipeline-parallel (PP) training job that is slower than expected.

Assumptions (state before acting): synchronous PP with a 1F1B schedule, P pipeline stages,
M microbatches per global batch, no interleaving unless confirmed from the launcher config.

Concrete mechanism. PP splits the model by layer into P stages; stage i sends the activation
tensor for a microbatch to stage i+1 and receives the gradient back on the backward pass.
Throughput loss comes from the pipeline bubble: in a synchronous 1F1B schedule the idle
fraction is (P-1)/(M+P-1) of ideal step time. With P=8 and M=8 that is 7/15 = 46.7% idle -
the job can look "GPU-bound" (high power draw on stage 0) while nearly half the wall clock is
warm-up/drain. Raising M to 64 drops the bubble to 7/71 = 9.9%. This formula is the first
thing to compute, before touching NCCL or the network.

Boundary condition. Increasing M is not free and stops helping at a hard limit: stage 0 must
hold up to P in-flight microbatch activation sets, so activation memory scales roughly with
min(M,P) x per-microbatch activations. Once microbatch size drops below the point where the
per-stage GEMMs saturate the tensor cores (commonly micro-batch x seq_len below a few
thousand tokens on A30/A100-class parts), per-microbatch efficiency falls faster than the
bubble shrinks, and total throughput regresses. So the bubble formula only applies while the
per-stage kernels remain compute-bound.

Falsifiable hypotheses, in order:
H1 bubble-dominated: measured step time is within ~10% of ideal_step x (M+P-1)/M. Test by
   doubling M at fixed global batch and predicting the new step time in advance.
H2 stage imbalance: one stage's forward time exceeds the mean by >15%, so every other stage
   waits on it. Test with per-stage forward/backward timers, not aggregate GPU utilization.
H3 link-bound P2P: send/recv time between adjacent stages is a significant share of step time.
   Test by placing stage boundaries on NVLink-connected pairs vs across the NIC and comparing.

Evidence to collect: launcher config (P, M, micro-batch size, interleaving degree), per-stage
step timers or a Nsight/torch profiler trace covering >=3 steady-state steps (never step 0),
NCCL_DEBUG=INFO topology dump to confirm which stage boundaries cross a node, and peak
per-stage allocated memory.

Rollback gate: any change to M, micro-batch size, or stage placement must hold the global
batch size and the optimizer hyperparameters constant. Revert if end-to-end tokens/s does not
improve by >=5% over a >=200-step window, or if loss over the first 200 steps diverges from
the pre-change run beyond its own step-to-step noise band. Do not judge from a single step.

Not measured here: any specific number for your cluster. All figures above are formula-derived
estimates and must be replaced with profiler measurements before you act on them."""

MOE_DEF = """Definition. Mixture-of-Experts (MoE) replaces a dense FFN block with N parallel expert FFNs
plus a small router (gate). For each token the router scores the experts and dispatches the
token to its top-k (commonly k=1 or 2). Only those k experts run, so the parameter count grows
with N while the FLOPs per token grow only with k.

Why it matters for LLM infrastructure. It decouples model capacity from per-token compute: an
N=64, k=2 MoE has roughly 32x the FFN parameters of the dense model at ~2x the FFN FLOPs. That
is a favorable trade when you are compute-limited, and an unfavorable one when you are
memory-capacity- or memory-bandwidth-limited, which is the usual case at inference.

Concrete mechanism (expert parallelism). Experts are sharded across devices, so dispatch is an
all-to-all: each rank sends its tokens to the ranks owning their chosen experts, runs the local
expert FFN, then a second all-to-all returns the results. Cost per MoE layer is therefore two
all-to-alls of size approximately (tokens_per_rank x k x d_model x dtype_bytes) each. The
collective is latency- and bisection-bandwidth-sensitive, so intra-node NVLink expert placement
behaves very differently from expert parallelism spanning nodes over RoCE/IB.

Boundary condition (capacity factor and token dropping). Each expert has a fixed buffer of
capacity_factor x (tokens x k / N) slots. Routing is data-dependent and skewed, so a hot expert
overflows and the surplus tokens are dropped - they bypass the FFN via the residual path and
silently lose their share of the computation. Raising the capacity factor removes the drops but
raises activation memory and all-to-all volume proportionally; lowering it raises drop rate.
There is no setting that is simultaneously drop-free and cheap under a skewed load, which is
why load-balancing auxiliary losses exist. The FLOP savings claim above holds only while the
drop rate is near zero and the router is well balanced.

Evidence needed before claiming an MoE win: per-expert token counts (to measure routing skew
and drop rate), the all-to-all time as a fraction of step time, achieved tokens/s, and peak
memory per rank. Do not use parameter count as a proxy for capability or FLOPs as a proxy for
speed.

Rollback gate: revert to the dense baseline if measured tokens/s per GPU does not beat dense at
equal quality, if drop rate exceeds a pre-agreed threshold (e.g. 1% of tokens) in steady state,
or if per-rank peak memory leaves less headroom than your longest-sequence safety margin."""

MOE_CONTRAST = """Contrast: MoE FFN vs the naive dense FFN it replaces.

Naive (dense) baseline. Every token passes through the same FFN with weight matrices of size
d_model x d_ff. Compute and parameters are coupled: to add capacity you must widen d_ff, and
every token then pays for it. Cost per token is roughly 2 x 2 x d_model x d_ff FLOPs, uniform
and perfectly predictable; there is no routing, no all-to-all, no load imbalance, and the
memory footprint is one copy of the FFN weights per model-parallel shard.

MoE. N expert FFNs plus a router; each token is sent to its top-k experts (typically k=1 or 2).
Parameters scale with N, per-token FLOPs with k. So capacity and compute are decoupled - the
whole point of the design.

Concrete mechanism that the dense version does not have: expert parallelism turns the FFN into
a communication step. Experts are sharded across ranks, so each MoE layer performs two
all-to-all collectives - one to dispatch tokens to expert owners, one to gather results back -
each moving about tokens_per_rank x k x d_model x dtype_bytes per rank. A dense FFN under
tensor parallelism instead performs a single all-reduce of the output activations, which maps
much better onto NVLink and degrades far more gracefully when the group spans nodes over
RoCE/InfiniBand. Replacing all-reduce with all-to-all changes which part of the fabric you are
stressing: all-to-all is bisection-bandwidth- and latency-sensitive and is the usual reason an
MoE that was fast on one node collapses on four.

Boundary condition where the contrast flips. The MoE advantage holds only while (a) routing is
balanced enough that the token drop rate at your capacity factor is near zero, and (b) the
all-to-all cost stays small relative to the expert GEMM time. Skewed routing forces you to
raise the capacity factor, which raises both activation memory and all-to-all volume; at small
per-rank token counts (short sequences, low batch, or heavy decoding) the expert GEMMs shrink
while the two collectives do not, so the dense FFN wins on wall clock despite doing more FLOPs.
Serving is the worst case: memory-bound decode must hold all N experts resident, so MoE costs
far more HBM per served token than the dense model it "saves compute" over.

Falsifiable claim to test rather than assert: at your target batch and sequence length, the MoE
layer's measured step time is lower than the dense layer's at matched quality. Evidence
required: per-expert token histograms (skew and drop rate), all-to-all time as a fraction of
step time from a profiler trace over >=3 steady-state steps, tokens/s per GPU, and peak memory
per rank for both variants under identical parallelism.

Rollback gate: revert to dense if tokens/s per GPU does not improve by >=10%, if steady-state
drop rate exceeds ~1% of tokens, or if peak per-rank memory removes the headroom needed for
your longest supported sequence."""

ANS = {
 'corpus-00300': PP_RUNBOOK,
 'corpus-00306': MOE_CONTRAST, 'corpus-00307': MOE_CONTRAST,
 'corpus-00308': MOE_CONTRAST, 'corpus-00309': MOE_CONTRAST,
}
for i in range(301,306):
    ANS['corpus-%05d'%i]=MOE_DEF

rows=[]
for sid in ['corpus-%05d'%i for i in range(300,310)]:
    r=src[sid]
    u=[m['content'] for m in r['messages'] if m['role']=='user'][0]
    a=[m['content'] for m in r['messages'] if m['role']=='assistant'][0]
    if sid=='corpus-00300':
        qd={'technical_correctness':4,'instruction_coverage':2,'operational_safety':3}
        risks=["Source answer states the mechanism but gives no bubble formula, so a reader cannot size the problem",
               "No boundary condition on how far microbatching can be pushed before activation memory or small-GEMM inefficiency dominates",
               "Runbook framing implies actionable steps but the source provides no ordered diagnostic procedure or rollback gate"]
        ev=["Launcher config: pipeline degree P, microbatch count M, micro-batch size, interleaving degree",
            "Per-stage forward/backward timers or profiler trace over >=3 steady-state steps",
            "Peak per-stage allocated memory before and after any change to M",
            "NCCL_DEBUG=INFO topology dump to identify which stage boundaries cross a node"]
        conf=0.79
    elif sid in ('corpus-00306','corpus-00307','corpus-00308','corpus-00309'):
        qd={'technical_correctness':4,'instruction_coverage':2,'operational_safety':3}
        risks=["Prompt asks for a contrast against a naive non-MoE implementation; the source answer never describes the dense baseline",
               "Omits that expert parallelism replaces all-reduce with all-to-all, the main cause of multi-node MoE regressions",
               "Does not state the regime where dense wins (small per-rank token counts, decode-time memory pressure), so it reads as an unconditional endorsement"]
        ev=["Per-expert token count histogram to quantify routing skew and drop rate",
            "Profiler trace over >=3 steady-state steps isolating all-to-all time vs expert GEMM time",
            "Tokens/s per GPU and peak per-rank memory for dense and MoE under identical parallelism",
            "Fabric topology: whether the expert-parallel group is intra-node NVLink or spans RoCE/IB"]
        conf=0.8
    else:
        qd={'technical_correctness':4,'instruction_coverage':2,'operational_safety':3}
        risks=["Mentions routing, capacity and all-to-all as keywords without explaining the token-dropping mechanism they cause",
               "No boundary condition: the compute-saving claim silently assumes near-zero drop rate and balanced routing",
               "Ignores the inference/serving case where all experts must stay resident, so MoE costs more HBM per served token"]
        ev=["Per-expert token counts and measured drop rate at the configured capacity factor",
            "All-to-all time as a fraction of step time from a profiler trace",
            "Peak per-rank memory and achieved tokens/s versus a dense baseline at matched quality"]
        conf=0.8
    rows.append({
      'source_id':sid,'teacher_lane':'teacher-B','teacher_model':'claude-opus-5-current',
      'calibration_status':'provisional','decision':'rewrite',
      'source_user':u,'source_assistant':a,'corrected_answer':ANS[sid],
      'quality_dimensions':qd,'risks':risks,'evidence_required':ev,'confidence':conf})

with open(OUT,'w') as f:
    for r in rows:
        f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('wrote',len(rows),OUT)
