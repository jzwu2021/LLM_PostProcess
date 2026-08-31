"""Measurement, metrics and evaluation-methodology mechanisms (topic: observability)."""
from __future__ import annotations

from core import Mechanism, Quant, Setting, fmt_int, gib, register


def q_percentile_merge(s: Setting) -> Quant:
    shards = max(s.gpu_count // max(s.tp, 1), 1)
    return Quant(
        label="why averaging per-replica tail latencies understates the fleet tail",
        steps=[
            f"The fleet has {shards} replica(s), each publishing its own high percentile",
            "The percentile of the combined distribution is not a function of the parts' percentiles",
            "Averaging them weights a lightly loaded replica equally with a saturated one",
            "Under uneven load the average therefore sits below the true combined tail",
        ],
        value=f"an average over {shards} replica tails is not the fleet tail",
        interpretation=(
            "Correct aggregation needs the raw observations or a mergeable summary with fixed bucket "
            "boundaries. Without one, the published tail is systematically optimistic exactly when "
            "load is skewed."),
    )


def q_sampling_bias(s: Setting) -> Quant:
    rate = 0.01
    return Quant(
        label="what head-based sampling does to a tail measurement",
        steps=[
            f"At a {rate:.0%} sampling rate, one request in {int(1 / rate)} is retained",
            f"To observe the p99 you need events beyond the 99th percentile to survive sampling",
            f"Those are one in 100 already, so after sampling they are one in {int(100 / rate)}",
            f"At {s.concurrency} concurrent requests the tail is populated by very few retained events",
        ],
        value=f"{rate:.0%} sampling leaves roughly one in {int(100 / rate)} requests informing the p99",
        interpretation=(
            "Uniform sampling preserves the shape of the body and destroys the resolution of the tail. "
            "Tail-aware or exhaustive collection is required for any percentile above the sampling "
            "rate's reciprocal."),
    )


def q_cardinality(s: Setting) -> Quant:
    labels = 6
    values = 20
    return Quant(
        label="the series count a few request labels generate",
        steps=[
            f"Suppose {labels} labels are attached to each latency metric",
            f"With about {values} distinct values each, the combinations are {values}^{labels}",
            f"That is {values ** labels:,} potential series from one metric",
            f"Across {s.gpu_count} devices reporting independently, multiply again",
        ],
        value=f"up to {values ** labels:,} series from one metric with {labels} labels",
        interpretation=(
            "Cardinality is multiplicative in labels, so adding one high-cardinality dimension can "
            "make a metrics system unusable. The failure appears as ingestion loss rather than as a "
            "metric that is simply larger."),
    )


def q_scrape_interval(s: Setting) -> Quant:
    interval = 60
    return Quant(
        label="what a coarse collection interval does to a short-lived event",
        steps=[
            f"Collection every {interval} s produces one sample per interval",
            f"A memory or queue spike lasting a few seconds falls entirely between samples",
            f"With a p99 objective of {s.slo_ms} ms, the events that breach it last milliseconds",
            f"So the metric that would explain the breach is never sampled during it",
        ],
        value=f"a {interval} s interval cannot observe events shorter than {interval} s",
        interpretation=(
            "Interval sets the shortest observable phenomenon. Diagnosing sub-second effects with "
            "minute-scale collection is not a matter of looking harder; the data does not exist."),
    )


def q_attribution(s: Setting) -> Quant:
    return Quant(
        label="why per-request timing cannot attribute cost under continuous batching",
        steps=[
            f"A step processes up to {s.concurrency} requests together",
            "Wall-clock time is a property of the step, not of any request within it",
            "Attributing step time to requests requires an allocation rule that is chosen, not measured",
            "Different rules assign the cost of one long prefill to entirely different requests",
        ],
        value=f"one step's time shared across up to {s.concurrency} requests by a chosen rule",
        interpretation=(
            "Per-request cost under batching is an accounting convention. Any conclusion that depends "
            "on it must state the rule, because a different rule supports a different conclusion."),
    )


def q_goodhart(s: Setting) -> Quant:
    return Quant(
        label="how repeated evaluation converts a measurement into a target",
        steps=[
            "A fixed evaluation set is used to accept or reject each candidate build",
            "Candidates that happen to suit the set survive; others are discarded",
            "After many rounds the surviving build is selected partly for fitting that set",
            "The set's score therefore rises while capability on unseen inputs does not",
        ],
        value="selection pressure accumulates on the evaluation set with every reuse",
        interpretation=(
            "The number of times a set has been used to make a decision is part of its result. A set "
            "reused for many rounds no longer measures what it originally measured."),
    )


def q_alert_precision(s: Setting) -> Quant:
    base = 0.001
    return Quant(
        label="why a highly accurate detector still pages mostly on false alarms",
        steps=[
            f"Suppose the condition occurs on {base:.1%} of evaluation windows",
            "A detector with 99% sensitivity and 99% specificity is considered good",
            f"Per 100000 windows: {int(100000 * base)} true events, "
            f"{int(100000 * (1 - base) * 0.01)} false positives",
            f"Precision = {int(100000 * base * 0.99)} / "
            f"({int(100000 * base * 0.99)} + {int(100000 * (1 - base) * 0.01)})"
            f" = {100000 * base * 0.99 / (100000 * base * 0.99 + 100000 * (1 - base) * 0.01) * 100:.1f}%",
        ],
        value=f"about {100000 * base * 0.99 / (100000 * base * 0.99 + 100000 * (1 - base) * 0.01) * 100:.1f}% "
              f"of pages would be true at a {base:.1%} base rate",
        interpretation=(
            "Alert quality is governed by base rate, not by detector accuracy. A rare condition needs "
            "far higher specificity than intuition suggests before paging is justified."),
    )


def q_baseline_band(s: Setting) -> Quant:
    return Quant(
        label="the run-to-run band that any claimed difference must exceed",
        steps=[
            "Run the identical configuration against itself several times",
            "Record the spread of the metric across those runs",
            f"On {s.gpu_count} devices, scheduling and batch composition vary between runs",
            "Any difference smaller than that spread is not distinguishable from repeating the same run",
        ],
        value="the self-comparison spread is the floor for any detectable difference",
        interpretation=(
            "This band costs a few extra runs and invalidates a large fraction of informal performance "
            "claims. It is the cheapest control available and the most frequently skipped."),
    )


def q_utilisation_meaning(s: Setting) -> Quant:
    return Quant(
        label="what the device utilisation counter actually reports",
        steps=[
            "The counter reports the fraction of sampled intervals with at least one active kernel",
            "It does not report how much of the device's arithmetic capacity that kernel used",
            "A memory-bound decode kernel occupies the device fully by this definition",
            f"So a bandwidth-limited service on {s.accel} can read near 100% while doing little arithmetic",
        ],
        value="occupancy of time, not of arithmetic capacity",
        interpretation=(
            "Using this counter as an efficiency measure conflates being busy with being productive. "
            "The roofline position, not the counter, says whether capability is being wasted."),
    )


def q_survivorship(s: Setting) -> Quant:
    return Quant(
        label="how failed requests disappear from a latency distribution",
        steps=[
            "Latency is recorded when a request completes",
            "Requests that time out or are rejected never record a completion",
            f"Under overload at {s.concurrency} concurrency, the slowest requests are the ones that fail",
            "So the recorded distribution improves precisely as the service degrades",
        ],
        value="the slowest requests are removed from the distribution by failing",
        interpretation=(
            "Latency must be read alongside completion rate. A tail that improves while success rate "
            "falls is a survivorship artefact rather than an improvement."),
    )


register(
    Mechanism(
        key="percentile_aggregation", topic="observability",
        title="averaging per-replica percentiles produces a number that is not a percentile",
        concepts=("percentiles", "aggregation", "metrics"),
        symptom="The published fleet tail latency is consistently better than what individual users experience.",
        chain="The percentile of a combined distribution is not a function of the parts' percentiles, so averaging per-replica tails weights a quiet replica equally with a saturated one and lands below the real combined tail whenever load is uneven.",
        metric="Tail computed by merging the underlying distributions, compared against the average of per-replica tails on the same data.",
        signature="The two figures diverge in proportion to load skew across replicas and agree when load is uniform.",
        confounders=(
            "Coarse histogram buckets near the tail, which introduce their own error in the merged figure.",
            "Clock skew between replicas, which misaligns the aggregation windows.",
            "Sampling applied before aggregation, which degrades the tail independently.",
        ),
        fixes=(
            "Aggregate with a mergeable summary such as a histogram with fixed bucket boundaries.",
            "Publish per-replica tails separately where merging is not available, rather than averaging them.",
            "Report the load skew alongside the tail so the size of the error is visible.",
        ),
        rollback="Restate historical thresholds under the corrected aggregation before switching, since alert thresholds tuned on the averaged figure are not valid under the merged one.",
        options=("publishing per-replica tails separately", "aggregating with a mergeable histogram"),
        tradeoff="whether the metrics pipeline can carry mergeable summaries rather than pre-computed percentiles",
        flip="bucket resolution near the tail becomes too coarse to be useful, at which point separate per-replica reporting is more honest than a merged figure with unstated error",
        falsifier="the merged and averaged tails agree under deliberately skewed load",
        wrong_claim="Fleet p99 is well inside the objective, so the latency objective is being met.",
        wrong_why="An average of per-replica p99 values is not the fleet p99, and under uneven load it sits below the real value, so the objective may be breached while the published figure says otherwise.",
        threshold="Require the tail to be computed from merged distributions rather than from averaged percentiles before it is used against an objective.",
        cost="An objective believed to be met while it is breached defers the capacity work that would have met it.",
        scaling="The error grows with replica count and with load skew, so it worsens as the fleet grows and as routing becomes less uniform.",
        quant=q_percentile_merge,
    ),
    Mechanism(
        key="sampling_destroys_tail", topic="observability",
        title="uniform trace sampling preserves the body of a distribution and destroys its tail",
        concepts=("tracing", "sampling", "percentiles"),
        symptom="Traces are collected at a low sampling rate and no example of the slow requests can ever be found.",
        chain="Retaining a fixed fraction of requests uniformly keeps that fraction of tail events too, so events beyond the high percentile become vanishingly rare in the retained set and the tail cannot be characterised from it.",
        metric="Number of retained traces beyond the percentile of interest, per collection window.",
        signature="The retained count beyond the target percentile is too small to characterise, while the body of the distribution is well populated.",
        confounders=(
            "Tail-based sampling already in use, which changes the retention rule and the analysis.",
            "Slow requests failing before completion, so they are absent for a different reason.",
            "Trace context propagation breaking, which loses spans regardless of sampling.",
        ),
        fixes=(
            "Switch to tail-based sampling so slow requests are retained preferentially.",
            "Retain all requests exceeding a latency threshold, independently of the sampling decision.",
            "Raise the rate only for the traffic class under investigation rather than globally.",
        ),
        rollback="Return to the previous rate if ingestion capacity is exceeded, since dropped spans produce a worse and less predictable bias than sampling.",
        options=("retaining all requests above a latency threshold", "switching to tail-based sampling"),
        tradeoff="whether the pipeline can defer the retention decision until the request completes",
        flip="the pipeline cannot buffer spans until completion, at which point a simple latency threshold on completed requests is the only workable rule",
        falsifier="the retained set contains enough events beyond the target percentile to characterise it",
        wrong_claim="We sample 1% of traces, which is statistically representative of the traffic.",
        wrong_why="It is representative of the body and not of the tail, because the events that define the tail are already rare and sampling multiplies their rarity.",
        threshold="Require enough retained events beyond the target percentile to characterise it before drawing any conclusion about the tail.",
        cost="Investigations proceed without evidence and conclude by guessing, which costs more than the retained storage would have.",
        scaling="Required retention grows as the target percentile rises, so a rate adequate for the median is useless for the p999.",
        quant=q_sampling_bias,
    ),
    Mechanism(
        key="label_cardinality", topic="observability",
        title="metric labels multiply, so one high-cardinality dimension can take down the pipeline",
        concepts=("cardinality", "metrics", "capacity_planning"),
        symptom="The metrics backend begins dropping data shortly after a new label was added to request metrics.",
        chain="Every combination of label values creates a distinct series, so cardinality is the product of the per-label value counts, and adding one dimension with many values multiplies the total rather than adding to it.",
        metric="Active series count per metric, tracked before and after any label change.",
        signature="Series count rises by roughly the factor of the new label's distinct value count, and ingestion loss begins at the backend's series limit rather than at a byte limit.",
        confounders=(
            "Traffic growth raising series count for a legitimate reason.",
            "A retention change altering the active series window.",
            "Another team adding labels at the same time, so attribution needs the change timeline.",
        ),
        fixes=(
            "Remove the high-cardinality label and carry that dimension in traces or logs instead.",
            "Bucket the label's values so its distinct count is bounded by design.",
            "Enforce a series budget per metric at the instrumentation layer rather than at the backend.",
        ),
        rollback="Remove the label immediately on ingestion loss rather than raising backend limits, since raising limits defers the same failure at higher cost.",
        options=("bucketing the label into a bounded set of values", "moving the dimension into traces instead of metrics"),
        tradeoff="whether the dimension is needed for aggregation or only for individual investigation",
        flip="the dimension turns out to be needed for aggregation across all requests, at which point bucketing rather than relocation is the only option",
        falsifier="series count is unchanged after the label was added",
        wrong_claim="It is one extra label, so it adds a small amount of metric volume.",
        wrong_why="Labels combine multiplicatively rather than additively, so one label with many distinct values multiplies the series count of every metric it is attached to.",
        threshold="Require the predicted series count after a label change to stay within the metric's series budget before the change ships.",
        cost="Ingestion loss removes observability during exactly the incidents that motivate having it.",
        scaling="Cardinality grows as the product of dimensions, so each added label is more expensive than the last.",
        quant=q_cardinality,
    ),
    Mechanism(
        key="collection_interval_floor", topic="observability",
        title="the collection interval sets the shortest phenomenon that can be observed at all",
        concepts=("sampling_interval", "metrics", "diagnosis"),
        symptom="A recurring failure leaves no trace in any dashboard, and every metric looks normal across the incident window.",
        chain="Metrics are sampled at a fixed interval, so any event shorter than that interval falls between samples, and the dashboard shows a normal value on both sides of a spike that was never observed.",
        metric="Collection interval compared against the duration of the phenomenon being investigated.",
        signature="The event's duration is below the collection interval, and higher-frequency collection on a single host reveals what the fleet-wide dashboards do not.",
        confounders=(
            "Averaging within the collection agent, which smooths spikes even at a fine interval.",
            "Rate-typed metrics reporting a change over the interval rather than an instantaneous value.",
            "Dashboards applying their own downsampling on top of the stored resolution.",
        ),
        fixes=(
            "Collect at a finer interval for the specific metric under investigation rather than fleet-wide.",
            "Record maximum-within-interval alongside the average so spikes survive aggregation.",
            "Use event-based instrumentation for phenomena shorter than any practical interval.",
        ),
        rollback="Return to the coarse interval once the investigation completes, since fine collection multiplies series volume across the fleet.",
        options=("recording maximum-within-interval alongside the average", "collecting the specific metric at a finer interval"),
        tradeoff="whether the phenomenon is short enough to require event-based rather than sampled observation",
        flip="the event is shorter than any practical sampling interval, at which point only event-based instrumentation can capture it",
        falsifier="the phenomenon lasts well beyond the collection interval and is still absent from the metric",
        wrong_claim="Nothing shows up in the metrics during the incident, so whatever happened was not resource related.",
        wrong_why="Sampled metrics cannot show events shorter than their interval, so absence of a signal is expected regardless of cause and carries no diagnostic information.",
        threshold="Match collection resolution to the duration of the phenomenon before concluding anything from an absent signal.",
        cost="Incidents recur while investigation proceeds against data that could not have contained the answer.",
        scaling="Fine-grained collection costs scale with fleet size, so the affordable interval lengthens as the fleet grows and the blind spot widens.",
        quant=q_scrape_interval,
    ),
    Mechanism(
        key="batched_cost_attribution", topic="observability",
        title="per-request cost under continuous batching is an accounting choice, not a measurement",
        concepts=("attribution", "continuous_batching", "cost_accounting"),
        symptom="Two teams derive different per-request costs from the same serving data and both derivations are internally consistent.",
        chain="A batched step's wall-clock time belongs to the step rather than to any request in it, so assigning it to individual requests requires an allocation rule, and different defensible rules produce different per-request costs.",
        metric="Step time and step composition recorded together, with the allocation rule stated explicitly.",
        signature="Per-request cost changes materially when the allocation rule changes while the underlying step data is unchanged.",
        confounders=(
            "Genuinely different request mixes between the two analyses.",
            "One analysis including queue wait and the other excluding it.",
            "Prefill and decode being attributed by different rules within the same analysis.",
        ),
        fixes=(
            "State the allocation rule with every per-request cost figure.",
            "Agree one rule across teams and apply it consistently rather than arguing about the result.",
            "Report step-level cost and composition directly where the decision permits it, avoiding attribution entirely.",
        ),
        rollback="Withdraw any per-request cost figure published without its allocation rule, since it cannot be compared against another figure.",
        options=("stating the allocation rule with every figure", "reporting step-level cost and composition directly"),
        tradeoff="whether the decision actually requires per-request granularity",
        flip="the decision genuinely needs per-request pricing, such as per-tenant billing, at which point a single agreed rule must be chosen and defended rather than avoided",
        falsifier="per-request cost is stable across defensible allocation rules",
        wrong_claim="Our telemetry reports the exact cost of each request, so we can price per request precisely.",
        wrong_why="Under batching the measured quantity is the step, and the per-request figure is produced by an allocation rule the telemetry chose silently, so its precision is a property of the rule rather than of the measurement.",
        threshold="Require the allocation rule to accompany any per-request cost used in a decision.",
        cost="Pricing or capacity decisions made on an unstated rule are not reproducible and are re-litigated at each review.",
        scaling="The ambiguity grows with batch size, since more requests share each step's time.",
        quant=q_attribution,
    ),
    Mechanism(
        key="evaluation_set_erosion", topic="observability",
        title="an evaluation set reused to make decisions stops measuring what it measured",
        concepts=("evaluation", "overfitting", "benchmark_hygiene"),
        symptom="Scores on the internal evaluation set improve steadily while user-visible quality does not change.",
        chain="Each round of accept-or-reject on a fixed set applies selection pressure toward that set, so the surviving candidate is chosen partly for fitting it, and the score rises without any corresponding gain on unseen inputs.",
        metric="Number of decisions the set has been used for, tracked alongside its score history.",
        signature="Score improves monotonically across rounds on the reused set while a freshly drawn set shows no corresponding movement.",
        confounders=(
            "Genuine capability improvement, which would also raise the fresh set's score.",
            "Set composition changing between rounds, which makes the series incomparable.",
            "Harness changes such as generation caps, which move scores for reasons unrelated to the model.",
        ),
        fixes=(
            "Hold a fresh set in reserve and use it only to confirm before release.",
            "Record how many decisions each set has informed and retire it past an agreed count.",
            "Rotate sets so no single one carries the whole selection history.",
        ),
        rollback="Treat a divergence between reused and fresh sets as invalidating the reused set rather than as a property of the fresh one.",
        options=("holding a fresh set in reserve for release confirmation", "rotating sets so none carries the whole history"),
        tradeoff="whether enough held-out material exists to keep a genuinely unused set available",
        flip="the reserve is exhausted and no unused material remains, at which point rotation only redistributes the erosion rather than preventing it",
        falsifier="a freshly drawn set moves in step with the reused one",
        wrong_claim="Our benchmark score has improved every quarter, which shows the model is getting better.",
        wrong_why="A set used to select between candidates accumulates selection pressure, so its score rises with reuse independently of capability, and only an unused set can distinguish the two.",
        threshold="Retire an evaluation set once it has informed more than an agreed number of accept-or-reject decisions.",
        cost="Releases justified by an eroded set ship changes whose real effect is unknown.",
        scaling="Erosion accumulates with decision count rather than with time, so rapid iteration exhausts a set faster than slow iteration.",
        quant=q_goodhart,
    ),
    Mechanism(
        key="alert_base_rate", topic="observability",
        title="alert precision is governed by the base rate, not by detector accuracy",
        concepts=("alerting", "base_rate", "precision"),
        symptom="A detector validated at high accuracy pages the on-call constantly and is almost always wrong.",
        chain="When the condition is rare, the small false-positive rate applied to the many normal windows produces more alerts than the high true-positive rate applied to the few real ones, so precision is low regardless of how accurate the detector is.",
        metric="Precision computed from the observed base rate together with the detector's sensitivity and specificity.",
        signature="Precision computed from the measured base rate matches the observed false-page rate, confirming rarity rather than detector quality as the cause.",
        confounders=(
            "A detector genuinely miscalibrated on this workload, which lowers specificity beyond its validation.",
            "Alert grouping or deduplication changing the apparent page count.",
            "The base rate differing between validation and production environments.",
        ),
        fixes=(
            "Raise specificity substantially rather than sensitivity, since the false-positive term dominates.",
            "Require a sustained condition across several windows before paging, which multiplies specificity.",
            "Route low-precision signals to a queue rather than to a page.",
        ),
        rollback="Return the detector to non-paging status if precision does not improve, rather than continuing to tune thresholds against an on-call rotation.",
        options=("requiring the condition to persist across several windows", "raising specificity directly"),
        tradeoff="whether the condition persists long enough that requiring several windows does not lose real events",
        flip="the real condition is short-lived, at which point requiring persistence discards true events and only specificity can be raised",
        falsifier="the observed false-page rate is far below what the base rate predicts",
        wrong_claim="The detector is 99% accurate in validation, so it will be a reliable alert.",
        wrong_why="Accuracy against a balanced validation set says nothing about precision against a rare production condition, where the false-positive count is set by the many normal windows.",
        threshold="Compute expected precision from the production base rate before a detector is allowed to page.",
        cost="Low-precision paging exhausts the on-call rotation and trains responders to ignore the signal, which is worse than having no alert.",
        scaling="The false-positive count grows with evaluation volume, so a detector that is tolerable on one service becomes unusable across a fleet.",
        quant=q_alert_precision,
    ),
    Mechanism(
        key="missing_baseline_band", topic="observability",
        title="a difference smaller than the run-to-run band is not a result",
        concepts=("benchmarking", "variance", "methodology"),
        symptom="A change is reported as a several-percent improvement based on one run before and one run after.",
        chain="Repeated runs of an identical configuration differ because scheduling, batch composition and placement vary, so any single-run comparison confounds the change with that variation, and differences inside the band are indistinguishable from repeating the same run twice.",
        metric="Spread of the metric across repeated runs of the unchanged configuration.",
        signature="The claimed difference falls inside the self-comparison spread, so the same difference appears between two runs of the identical build.",
        confounders=(
            "A genuine effect larger than the band, which the same procedure would confirm.",
            "Systematic drift across the measurement period, which repetition detects but a single pair does not.",
            "Warm-up contaminating the first run of each pair.",
        ),
        fixes=(
            "Run the unchanged configuration against itself several times and publish the band with every result.",
            "Interleave the arms rather than running them in sequence, so drift affects both equally.",
            "Report the number of runs and the spread rather than a single point difference.",
        ),
        rollback="Withdraw any claim whose difference falls inside the measured band rather than defending it with additional single runs.",
        options=("publishing the self-comparison band with every result", "interleaving the arms to cancel drift"),
        tradeoff="whether the observed difference exceeds the variation the environment produces on its own",
        flip="drift over the measurement period becomes larger than the band itself, at which point interleaving matters more than repetition count",
        falsifier="the difference is several times the measured band",
        wrong_claim="Throughput went from 1,020 to 1,065 tokens per second, so the change gives about 4%.",
        wrong_why="Without the run-to-run band those two numbers may be two samples from the same distribution, and on a shared cluster a 4% spread between identical runs is common.",
        threshold="Require any claimed difference to exceed the measured self-comparison band before it is reported.",
        cost="Changes adopted on noise accumulate in the codebase and are never removed, because the evidence to remove them is equally weak.",
        scaling="The band widens with cluster sharing and with batch-composition variability, so busier environments need more repetitions for the same claim.",
        quant=q_baseline_band,
    ),
    Mechanism(
        key="utilisation_counter_meaning", topic="observability",
        title="the device utilisation counter measures occupancy of time, not use of capability",
        concepts=("utilisation", "roofline", "metrics"),
        symptom="Device utilisation reads near its maximum while achieved arithmetic throughput is a small fraction of the device's capability.",
        chain="The counter reports the fraction of sampled intervals in which any kernel was resident, so a memory-bound kernel that keeps the device busy while performing little arithmetic registers as fully utilised.",
        metric="Achieved arithmetic throughput and achieved memory bandwidth, reported alongside the utilisation counter.",
        signature="Utilisation is high while achieved arithmetic throughput is low and achieved bandwidth is near the device's measured peak.",
        confounders=(
            "Small kernels with launch gaps, which lower the counter without lowering capability use.",
            "Multi-process sharing, which makes the counter reflect combined occupancy.",
            "Profiling overhead altering both the counter and the throughput measurement.",
        ),
        fixes=(
            "Report achieved bandwidth and arithmetic throughput instead of relying on the occupancy counter.",
            "Locate the workload on a roofline so the binding resource is explicit.",
            "Use the counter only to detect idleness, which is the question it can actually answer.",
        ),
        rollback="Stop optimising toward the counter if achieved throughput does not move with it, since the counter is not the objective.",
        options=("reporting achieved bandwidth and arithmetic throughput", "locating the workload on a roofline"),
        tradeoff="whether the question being asked is about idleness or about efficiency",
        flip="the question becomes whether devices are idle rather than whether they are efficient, at which point the occupancy counter is exactly the right instrument",
        falsifier="achieved arithmetic throughput tracks the utilisation counter across the load range",
        wrong_claim="GPU utilisation is at 95%, so the hardware is being used efficiently and we need more of it.",
        wrong_why="The counter reports that a kernel was resident, not that the device's arithmetic capability was consumed, and a bandwidth-bound decode phase saturates the counter while using a small share of that capability.",
        threshold="Require achieved bandwidth or arithmetic throughput, not the occupancy counter, before concluding that capability is exhausted.",
        cost="Capacity purchased on an occupancy signal buys arithmetic the workload is structurally unable to use.",
        scaling="The gap between occupancy and capability use widens as decode dominates, so it grows with generation length and with concurrency.",
        quant=q_utilisation_meaning,
    ),
    Mechanism(
        key="latency_survivorship", topic="observability",
        title="failed requests leave the latency distribution, so the tail improves as the service degrades",
        concepts=("survivorship_bias", "latency", "slo"),
        symptom="Tail latency improved during an incident while users reported the service as unusable.",
        chain="Latency is recorded on completion, and under overload the slowest requests are the ones that time out or are rejected, so they never contribute a sample and the recorded distribution improves precisely as service quality falls.",
        metric="Latency distribution read together with completion rate over the same window.",
        signature="The tail improves while completion rate falls, and the improvement magnitude tracks the share of requests that failed.",
        confounders=(
            "A genuine latency improvement, which would not be accompanied by falling completion rate.",
            "Load shedding by design, which produces the same signature but as an intended policy.",
            "Client-side timeouts shortening, which removes slow requests for a reason outside the service.",
        ),
        fixes=(
            "Publish latency and completion rate as a single joint objective rather than as separate metrics.",
            "Record a bounded penalty value for failed requests so they remain in the distribution.",
            "Alert on the combination rather than on either metric alone.",
        ),
        rollback="Discard any latency-based conclusion drawn over a window in which completion rate fell, and re-measure once completion is restored.",
        options=("publishing latency and completion rate as one joint objective", "recording a penalty value for failed requests"),
        tradeoff="whether a synthetic penalty value distorts the distribution more than the omission does",
        flip="the penalty value dominates the tail and hides real latency variation, at which point the joint objective is the cleaner presentation",
        falsifier="completion rate is flat across the window in which the tail improved",
        wrong_claim="The p99 latency dropped during the incident, so the latency path was not the problem.",
        wrong_why="The requests that would have populated the tail failed instead of completing, so their absence improved the statistic while representing the worst possible user outcome.",
        threshold="Never interpret a latency change without the completion rate over the same window.",
        cost="An incident misdiagnosed as unrelated to latency continues while the responder investigates elsewhere.",
        scaling="The bias grows with the failure rate, so it is strongest during the most severe incidents where accurate measurement matters most.",
        quant=q_survivorship,
    ),
)
