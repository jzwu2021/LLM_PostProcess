import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{EXP}/results/train-batch-0257.jsonl"
START, END = 2560, 2570

corpus = [json.loads(l) for l in open(CORPUS) if l.strip()]
src = corpus[START:END]

STANCES = [
 ("STANCE 141 - Publish the step-time distribution with uncertainty on its tail quantiles, because a mean step time is precisely the statistic that imbalance does not move.",
  """Mechanism. Expert imbalance delays the steps in which a hot expert overflows or a rank arrives late, leaving the majority of steps unaffected. The mean therefore shifts only slightly while the upper quantiles shift a great deal. Reporting a mean, or a p99 without an interval, makes an intervention look ineffective or effective according to sampling noise, and tail quantiles estimated from short windows are far noisier than practitioners generally assume.

Falsifiable hypothesis. H1: the intervention shifts p99 step time by more than the width of the p99 confidence interval computed from the observed window, while leaving mean step time inside its own interval. Falsified if p99 moves no more than its interval width, which means the window is too short to resolve the claimed effect regardless of the point estimate.

Metrics. Full step-time distribution with p50, p90, p99 and p999, bootstrap confidence intervals on each quantile, number of steps observed, mean step time reported alongside for contrast, per-expert load dispersion, token-drop rate and all-to-all time. Quantiles and their intervals are MEASURED by bootstrap over the recorded steps; any quantile quoted without its interval is an ESTIMATE of unknown precision.

Controlled experiment. Collect step times over a window sized so the target quantile has a pre-declared interval width, computing the required step count in advance rather than reporting whatever the available window yields. Then run control and intervention arms over equal-length windows and compare quantiles with their intervals rather than comparing point estimates.

Confounders. Steps are not independent, so a naive bootstrap understates interval width and a block bootstrap respecting autocorrelation is required. Mixing prefill and decode steps produces a bimodal distribution whose quantiles are not interpretable as a single population. Warm-up steps at the start of a run inflate the upper tail.

Rollback criteria. Withdraw any tail-latency claim whose quantile interval overlaps the comparator, and extend the window rather than restating the point estimate. Do not set an alerting threshold on a quantile whose interval is wider than the threshold's own margin, since such an alert will fire on noise and be silenced."""),
 ("STANCE 142 - Decompose end-to-end latency before attributing the tail to routing, because queueing and admission usually contribute more than the collective does.",
  """Mechanism. A request's latency is the sum of time spent queued for admission, time in prefill, time in decode steps, and any preemption gaps. Expert imbalance affects only part of the step time inside prefill and decode. When utilisation is high, queueing dominates and a large improvement in step time produces a small improvement in end-to-end latency. Attributing the user-visible tail to routing without the decomposition risks optimising a term that is not binding.

Falsifiable hypothesis. H1: the collective and expert-compute terms together account for a minority of end-to-end p99 latency at the current operating point, so eliminating imbalance entirely would leave most of the tail in place. Falsified if those terms dominate the tail, which makes routing and placement the correct first target.

Metrics. End-to-end p99 latency decomposed into queue wait, prefill compute, expert compute, collective, decode compute and preemption gaps, each with its share of the tail; utilisation and admitted concurrency; per-expert dispersion; token-drop rate. The decomposition is MEASURED from per-request spans that sum to the observed latency, and a decomposition that does not sum to the total is rejected rather than normalised.

Controlled experiment. Instrument per-request spans covering every phase and verify that their sum reconciles with the independently measured end-to-end latency before analysing shares. Then repeat the decomposition at two utilisation levels, since the queueing share grows sharply near saturation and a single operating point cannot show that.

Confounders. Overlapped phases cannot be summed naively and require either exclusive attribution or an explicit overlap term. Client-side time including connection setup is outside the server spans and can dominate for short requests. Preemption gaps are frequently unlogged and appear as inflated decode time.

Rollback criteria. Do not fund routing or placement work while the decomposition shows queueing dominant; record the decomposition as the justification for whichever term is targeted so a later re-measurement can falsify the choice. If an intervention improves its target term without moving end-to-end p99, report it as such rather than claiming a latency win."""),
 ("STANCE 143 - Distinguish imbalance across experts from imbalance across data-parallel replicas, because uneven traffic between replicas produces the same tail with a different cause.",
  """Mechanism. A deployment typically runs several data-parallel replicas, each holding the full expert set or its own expert shard group. If the load balancer distributes requests unevenly, or if replicas differ in co-tenancy or hardware, one replica will show longer steps regardless of how well experts are balanced within it. The per-expert histogram aggregated across replicas can look uniform while a single replica is saturated, and the tail is set by whichever replica served the slow request.

Falsifiable hypothesis. H1: between-replica variation in step time exceeds within-replica variation attributable to expert imbalance, locating the dominant skew at the replica level. Falsified if replicas are statistically interchangeable and all skew is within-replica, which confirms routing and placement as the correct target.

Metrics. Step time and per-expert dispersion per replica rather than pooled, request share per replica, per-replica co-tenancy and hardware identifiers, between-replica variance versus within-replica variance, end-to-end p99 by replica, and token-drop rate by replica. Per-replica figures are MEASURED by stamping the replica identifier on every request and step record.

Controlled experiment. Stamp replica identity and recompute every balance and latency metric grouped by replica, then test whether replicas are exchangeable by comparing their distributions against each other before any pooled analysis is performed. Where a replica is an outlier, compare its hardware and co-tenancy against the others before considering any routing explanation.

Confounders. Load balancers with session affinity send correlated traffic to the same replica, so replica differences may reflect traffic composition rather than replica health. Autoscaling replaces replicas mid-window, so the identifier set is not stable. A replica that recently started is still warming and will look slow for reasons unrelated to balance.

Rollback criteria. Do not apply a fleet-wide routing or placement change to correct what the per-replica analysis shows to be a single-replica problem; drain and investigate that replica instead. Any load-balancer change made in response must be revertible by configuration and must be checked for fairness across tenants, since rebalancing traffic can concentrate one tenant on one replica."""),
 ("STANCE 144 - Choose a baseline that makes the comparison meaningful: a dense model with matched active parameters, not the MoE's own theoretical FLOPs.",
  """Mechanism. Sparse MoE architectures are adopted because they raise total parameters while keeping active parameters per token roughly constant. Efficiency claims stated against total parameters flatter the design, and claims stated against theoretical FLOPs ignore the collective, the padding and the routing overhead that the dense model does not pay. The only comparison that supports a deployment decision is against the dense alternative that would otherwise be served, at matched quality or matched active parameters.

Falsifiable hypothesis. H1: at matched active parameters, the MoE deployment achieves lower tokens per GPU-second than the dense comparator once collective time, padding and routing overhead are counted, even though its theoretical FLOPs per token are similar. Falsified if the MoE matches or exceeds the dense comparator on that measure, which supports the architecture on serving grounds rather than only on quality grounds.

Metrics. Tokens per GPU-second and GPU-seconds per useful token for both models, active parameters per token, total parameters, collective time share, useful-token fraction, p99 step time, end-to-end p99 latency, and fixed eval-set quality for both. All throughput figures are MEASURED on the same hardware, batch policy and serving stack; theoretical FLOPs figures may be reported only as an ESTIMATE and never as the efficiency claim.

Controlled experiment. Serve both models on identical hardware with identical batching, admission and decode configuration, replaying the same request stream, and report the quality-versus-throughput pair for each rather than either alone. Because the two differ in quality, the comparison must be presented as a frontier and not as a single winner.

Confounders. The dense comparator may be less optimised in the current stack, so a throughput gap can reflect engineering effort rather than architecture. Quantisation support commonly differs between the two. Memory footprint differs greatly, so achievable batch size differs and must be reported rather than equalised artificially.

Rollback criteria. Withdraw any efficiency claim stated against total parameters or theoretical FLOPs and restate it against the dense comparator. If a deployment decision was already taken on such a claim, re-derive it before scaling the fleet, and retain the dense artifact as a deployable fallback until the comparison is complete."""),
 ("STANCE 145 - Treat routing as numerics-sensitive: tokens near the gate decision boundary flip between kernels and precisions, so the same weights do not imply the same assignment.",
  """Mechanism. The router picks the top-k experts by comparing gate logits. For tokens whose top candidates are close, a difference of a few units in the last place decides the assignment. Attention and matmul kernels differ in accumulation order across backends, batch sizes and precisions, so the same checkpoint can route a small fraction of tokens differently on different builds. Those flipped tokens change per-expert counts and, at capacity boundaries, change which tokens are dropped.

Falsifiable hypothesis. H1: a non-trivial fraction of tokens have a top-1 to top-2 gate margin small enough to flip under the numeric differences between two deployed builds, and the resulting assignment differs between them on identical input. Falsified if assignments are identical across builds, which means the margin distribution is safely away from the boundary and routing can be treated as build-invariant.

Metrics. Distribution of the top-1 minus top-2 gate margin, fraction of tokens below a build-difference threshold, measured assignment disagreement rate between builds on identical input, resulting per-expert count differences, drop-rate difference, and fixed eval-set quality per build. The disagreement rate is MEASURED by running the same input through both builds and comparing assignments directly.

Controlled experiment. Run one fixed token stream through the two builds at temperature zero, capture per-token assignments, and compute the disagreement rate and the margin distribution from the same capture. Then check whether disagreements concentrate at capacity boundaries, since a flip only matters operationally when it changes a drop decision.

Confounders. Batch size changes accumulation order in some kernels, so a disagreement observed at one batch size may vanish at another and batch size must be pinned. Non-determinism within a single build masquerades as cross-build disagreement unless within-build repeatability is established first. Mixed-precision autocasting policies differ across frameworks.

Rollback criteria. Establish within-build repeatability before reporting any cross-build disagreement, and treat a build that is not repeatable with itself as unsuitable for routing analysis until fixed. Do not carry a placement or capacity tuning derived on one build to another without re-measuring the assignment distribution, and record the build identifier with every tuning artifact."""),
 ("STANCE 146 - Evaluate alternative capacity and placement policies offline against recorded routing scores, because the comparison needs no serving and removes every serving confound.",
  """Mechanism. Once per-token gate scores are recorded for a window, the consequences of a different capacity factor or a different expert-to-device mapping are a deterministic function of that record. Which tokens would have been dropped, how many bytes would have crossed each link, and how loaded each device would have been can all be computed without running the model. This turns an expensive serving experiment into a cheap offline sweep and eliminates queueing, cache warmth and co-tenancy from the comparison entirely.

Falsifiable hypothesis. H1: offline-predicted drop rates and per-device loads for a candidate policy match the values realised when that policy is actually served, within a pre-declared tolerance. Falsified if prediction and realisation diverge, which identifies a serving-side effect the offline model omits and bounds how far offline sweeps can be trusted.

Metrics. Predicted and realised token-drop rate, per-device load, inter-node and intra-node bytes for each candidate policy; prediction error per metric; the size of the recorded score window; and p99 step time for the realised runs. Predictions are ESTIMATE by construction and must be labelled as such until the validation run makes the corresponding realised figures MEASURED.

Controlled experiment. Record gate scores for a representative window, sweep candidate policies offline, then serve the top candidate and the current policy and compare realised against predicted values. Validate the offline model on the current policy first, where the realised values are already known, before trusting its ranking of alternatives.

Confounders. Batch composition under a different policy would differ in production but is fixed in the recording, so the offline model assumes composition is policy-independent, which is false when capacity affects scheduling. Recorded scores are tied to one checkpoint and build. Score recording is expensive and is usually sampled, so the window may not cover the tail.

Rollback criteria. Do not deploy a policy on offline evidence alone; require the validation run, and abandon the offline model for policies where its prediction error exceeds the tolerance rather than adjusting the tolerance. The recorded score window must be re-collected after any checkpoint or build change, since a stale recording silently invalidates every prediction derived from it."""),
 ("STANCE 147 - Compute an effective sample size for step-level statistics, because consecutive steps share batch composition and are not independent draws.",
  """Mechanism. A request occupies many consecutive steps, and the set of co-resident requests changes slowly relative to the step rate. Expert load in one step is therefore strongly predictive of the next, and a window of ten thousand steps may contain only a few hundred effectively independent observations. Confidence intervals and significance tests built on the raw step count will be far too narrow, and small differences will be declared real.

Falsifiable hypothesis. H1: the autocorrelation of per-step expert dispersion at short lags is high enough that the effective sample size is a small fraction of the raw step count, so intervals computed on raw counts are materially too narrow. Falsified if autocorrelation decays within a step or two and the effective size approaches the raw count, which validates the simpler analysis.

Metrics. Autocorrelation function of per-step dispersion and of step time, integrated autocorrelation time, effective sample size against raw step count, interval widths computed both ways, and the minimum detectable difference in p99 step time at the realised effective size. The autocorrelation and effective size are MEASURED from the recorded series rather than assumed.

Controlled experiment. Estimate the autocorrelation on a control window with no intervention, derive the effective sample size, and use it to size the comparison window before running the arms. Then verify by running two control windows against each other and confirming the false-positive rate matches the nominal level under the corrected intervals.

Confounders. Autocorrelation varies with load, so an estimate from a quiet period understates it at peak. Long requests induce longer-range correlation than short ones, so the mix determines the correlation structure. Non-stationarity within the window inflates apparent autocorrelation and must be checked separately from genuine short-lag dependence.

Rollback criteria. Reject any step-level significance claim computed on raw step counts and recompute with the corrected effective size before it is used for a decision. If a rollout was gated on such a claim, treat it as inconclusive and re-run at a window sized against the measured effective sample size."""),
 ("STANCE 148 - Add an explicit capacity-overflow counter and alert, because dropping tokens raises no error and will otherwise be discovered only through a quality complaint.",
  """Mechanism. When an expert's capacity is exceeded the surplus tokens are discarded silently. No exception is raised, no request fails, and throughput counters look normal. The only visible consequence is a change in model output, which surfaces as a diffuse quality complaint days or weeks later with no obvious link to a capacity or traffic change. This is a silent failure mode and it requires a dedicated counter, a threshold and a route, exactly as any other correctness-affecting condition would.

Falsifiable hypothesis. H1: the current deployment emits no metric from which a non-zero token-drop rate could be detected, so an overflow event would be invisible to monitoring. Falsified if an existing metric moves detectably when drops occur, which would mean the condition is already observable and only needs a threshold.

Metrics. Token-drop count and rate per expert per layer, drop-rate alert threshold and its route, time from a seeded drop event to alert delivery, false-alert rate over a trailing window, and the correlation between drop rate and fixed eval-set quality. The detection latency is MEASURED by seeding a drop condition, not asserted from the metric's existence.

Controlled experiment. Seed an overflow by lowering capacity on a canary replica and measure whether existing monitoring surfaces anything, then add the dedicated counter and repeat, measuring delivery and acknowledgement. Separately, sweep the drop rate and measure eval-set quality at each level so the alert threshold is set against a measured quality effect rather than an arbitrary round number.

Confounders. Per-expert per-layer counters are high cardinality and can be prohibitively expensive, so aggregation choices determine what remains detectable. Drops concentrated in one layer may matter far more than the same rate spread across layers. Quality effects of low drop rates may be below the resolution of the eval set, making the threshold unanchored.

Rollback criteria. Do not lower capacity factor for throughput gains until the counter and alert are deployed and their route tested, since the resulting condition would be undetectable. If the alert's false-positive rate exceeds its ceiling, demote it to advisory by configuration rather than allowing it to be silenced informally."""),
 ("STANCE 149 - Rule out a hardware straggler before rebalancing anything, because one degraded device produces exactly the signature attributed to expert imbalance.",
  """Mechanism. In a synchronous collective, the step time is set by the slowest participant. A device that is thermally throttled, has degraded ECC-corrected memory, sits behind a flapping link or shares a host with a noisy neighbour will arrive late at every all-to-all. Aggregated per-expert histograms will look unremarkable while step time and rank wait are elevated, and the natural but incorrect reading is that experts are unbalanced. The hardware hypothesis is cheaper to test and must be excluded first.

Falsifiable hypothesis. H1: the elevated rank wait time is concentrated on a stable subset of devices that persists across placements and workloads, indicating hardware rather than routing. Falsified if the late-arriving rank moves with the expert placement, which confirms load rather than hardware as the cause.

Metrics. Per-rank arrival time at the collective over a long window, identity stability of the latest rank across runs and placements, per-device clock frequency and thermal throttling counters, ECC corrected-error counts, link flap and retransmission counters, per-device achieved FLOPs on a fixed microbenchmark, and p99 step time. Device counters are MEASURED from the driver and fabric rather than inferred from step timing.

Controlled experiment. Run an identical dense microbenchmark on every device with no routing involved and compare achieved throughput, which isolates hardware from any expert-related effect. Then permute the expert placement and observe whether the late rank follows the placement or stays with the device.

Confounders. Co-tenancy on the same host varies over time and mimics intermittent hardware degradation. Thermal throttling is load-dependent, so a device tests healthy at idle and degrades under sustained load, meaning the microbenchmark must be sustained. Firmware and driver versions can differ across a fleet and change performance without any physical fault.

Rollback criteria. Drain and replace a device identified as a persistent straggler rather than compensating for it with placement, since placement tuned around a faulty device becomes wrong the moment it is repaired. Any placement change made before hardware was excluded must be reverted and re-derived once the fleet is homogeneous."""),
 ("STANCE 150 - Measure incast at the fabric, because an all-to-all makes every rank a simultaneous receiver and the switch buffer, not the link rate, often sets the limit.",
  """Mechanism. In an all-to-all every rank sends to every other rank at once, so each receiver's port is targeted by many senders simultaneously. When the aggregate arriving rate exceeds the egress port capacity, packets queue in switch buffers and, once those fill, are marked or dropped. The result is congestion control backing off and effective bandwidth far below the nominal link rate. This limit is invisible in host-side timers and appears only as an unexplained gap between achieved and theoretical transfer time.

Falsifiable hypothesis. H1: during the all-to-all, switch buffer occupancy and congestion marking rise sharply and achieved bandwidth falls materially below the link rate, identifying incast rather than volume as the binding constraint. Falsified if achieved bandwidth approaches the link rate with negligible marking, which points back to volume and therefore to placement and capacity.

Metrics. Achieved versus theoretical bandwidth per link during the collective, switch buffer occupancy, congestion notification and pause-frame counts, retransmission counts, per-rank transfer time, arrival-time spread, and p99 step time. Fabric counters are MEASURED from switch telemetry; host-side timing alone cannot distinguish incast from insufficient volume.

Controlled experiment. Collect switch telemetry synchronised with the collective's timeline on a fixed replayed stream, then reduce the number of simultaneous senders by staging the exchange in phases and observe whether achieved bandwidth improves at constant total bytes. Constant bytes with improved throughput demonstrates incast rather than volume as the constraint.

Confounders. Other tenants sharing the fabric contribute to the same buffers, so the measurement must record neighbour activity or be taken on a quiesced fabric. Congestion control settings such as explicit notification thresholds change the marking rate independently of load. Switch telemetry sampling intervals are often coarse relative to the collective's duration and can miss the burst entirely.

Rollback criteria. Do not request a fabric upgrade on host-side timing alone; require the telemetry that distinguishes incast from volume. Any congestion-control or staging change must be revertible by configuration and validated for its effect on other tenants sharing the fabric, since tuning for this workload can degrade theirs."""),
]

DECISIONS = ["rewrite"] * 10

QD = [
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
 (2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),
]

CONF = [0.73,0.74,0.71,0.72,0.7,0.71,0.72,0.74,0.73,0.7]

RISKS = [
 ["Source answer asks for tail latency without requiring the distribution or uncertainty on its quantiles.",
  "Steps are autocorrelated, so a naive bootstrap understates the interval width on tail quantiles.",
  "Mixing prefill and decode steps produces a bimodal distribution whose pooled quantiles are not interpretable."],
 ["Source answer targets tail latency without decomposing it into queueing, compute and collective terms.",
  "Overlapped phases cannot be summed naively and need explicit attribution or an overlap term.",
  "Unlogged preemption gaps appear as inflated decode time and misdirect attribution."],
 ["Source answer measures expert load without separating within-replica skew from between-replica skew.",
  "Session affinity sends correlated traffic to the same replica, so replica differences may reflect composition rather than health.",
  "Autoscaling replaces replicas mid-window, so the replica identifier set is not stable."],
 ["Source answer compares routing and capacity policies without fixing the baseline the deployment decision rests on.",
  "A dense comparator may be less optimised in the current stack, so a gap can reflect engineering effort rather than architecture.",
  "Memory footprint differs greatly between the two, so achievable batch size differs and cannot be equalised artificially."],
 ["Source answer assumes a fixed checkpoint implies a fixed routing assignment.",
  "Batch size changes accumulation order in some kernels, so cross-build disagreement is batch-size dependent.",
  "Within-build non-determinism masquerades as cross-build disagreement unless repeatability is established first."],
 ["Source answer proposes comparing routing and capacity policies in serving, where every serving confound applies.",
  "The offline model assumes batch composition is policy-independent, which fails when capacity affects scheduling.",
  "Recorded gate scores are tied to one checkpoint and build and are usually sampled, so the tail may be uncovered."],
 ["Source answer treats step-level observations as independent when computing evidence for a difference.",
  "Autocorrelation varies with load, so an estimate from a quiet period understates it at peak.",
  "Non-stationarity within a window inflates apparent autocorrelation and must be checked separately."],
 ["Source answer lists token dropping as a measurement without requiring it to be alertable.",
  "Per-expert per-layer counters are high cardinality, so aggregation choices decide what remains detectable.",
  "Quality effects of low drop rates may fall below eval-set resolution, leaving the alert threshold unanchored."],
 ["Source answer attributes tail latency to routing without excluding a degraded device first.",
  "Thermal throttling is load-dependent, so a device tests healthy at idle and degrades only under sustained load.",
  "Placement tuned around a faulty device becomes wrong as soon as the device is repaired."],
 ["Source answer measures all-to-all time without distinguishing insufficient bandwidth from receiver-side incast.",
  "Other tenants share the same switch buffers and contribute to the observed congestion.",
  "Switch telemetry sampling is often coarse relative to the collective's duration and can miss the burst."],
]

EVID = [
 ["Full step-time distribution with p50, p90, p99 and p999 and block-bootstrap confidence intervals respecting autocorrelation.",
  "Pre-declared window size derived from the target interval width, with prefill and decode steps separated before quantiles are computed."],
 ["Per-request spans covering queue wait, prefill, expert compute, collective, decode and preemption that reconcile with measured end-to-end latency.",
  "Decomposition repeated at two utilisation levels so the growth of the queueing share near saturation is visible."],
 ["Replica identifier stamped on every request and step record, with balance and latency metrics recomputed per replica.",
  "Exchangeability check between replica distributions plus hardware and co-tenancy comparison for any outlier replica."],
 ["Tokens per GPU-second and GPU-seconds per useful token for the MoE and a matched-active-parameter dense comparator on identical hardware and batching.",
  "Quality-versus-throughput frontier reported for both, with achievable batch size and quantisation support recorded rather than equalised."],
 ["Top-1 minus top-2 gate margin distribution and measured assignment disagreement rate between builds on identical input at pinned batch size.",
  "Within-build repeatability established first, with disagreement concentration at capacity boundaries reported separately."],
 ["Recorded per-token gate scores for a representative window with the checkpoint and build identifiers attached.",
  "Offline model validated against the current policy's already-known realised values before its ranking of alternatives is used."],
 ["Autocorrelation function and integrated autocorrelation time of per-step dispersion measured on a control window.",
  "Effective sample size used to size the comparison window, verified by a control-versus-control false-positive check."],
 ["Per-expert per-layer token-drop counters with a seeded drop event measuring end-to-end alert delivery and acknowledgement.",
  "Drop-rate sweep against fixed eval-set quality so the alert threshold is anchored to a measured quality effect."],
 ["Per-rank arrival times over a long window with the identity of the latest rank tracked across placements and workloads.",
  "Sustained dense microbenchmark per device with driver and fabric counters for throttling, ECC errors and link flaps."],
 ["Switch telemetry for buffer occupancy, congestion notifications and pause frames synchronised with the collective timeline.",
  "Achieved-versus-theoretical bandwidth compared at constant total bytes with the number of simultaneous senders reduced by phased exchange."],
]

rows = []
for i, s in enumerate(src):
    m = {x["role"]: x["content"] for x in s["messages"]}
    head, body = STANCES[i]
    tc, ic, os_ = QD[i]
    rows.append({
        "source_id": s["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": DECISIONS[i],
        "source_user": m["user"],
        "source_assistant": m["assistant"],
        "corrected_answer": head + "\n\n" + body,
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": ic,
            "operational_safety": os_,
        },
        "risks": RISKS[i],
        "evidence_required": EVID[i],
        "confidence": CONF[i],
    })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", OUT, len(rows))
