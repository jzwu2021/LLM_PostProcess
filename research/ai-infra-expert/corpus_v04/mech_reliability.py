"""Reliability, rollout and failure-handling mechanisms (topic: reliability)."""
from __future__ import annotations

from core import Mechanism, Quant, Setting, fmt_int, gib, register


def q_correlated_failure(s: Setting) -> Quant:
    replicas = max(s.gpu_count // max(s.tp, 1), 1)
    return Quant(
        label="why independent-failure arithmetic overstates availability",
        steps=[
            f"With {replicas} replica(s), independent failures would make simultaneous loss rare",
            "Replicas share an image, a config rollout, a model artifact and often a rack",
            "A fault in any shared element fails them together rather than independently",
            f"So the effective independent count is below {replicas}, sometimes one",
        ],
        value=f"{replicas} nominal replica(s), fewer independent failure domains",
        interpretation=(
            "Availability computed by multiplying per-replica failure probabilities assumes an "
            "independence the deployment does not have. The shared elements are the real domain "
            "count."),
    )


def q_blast_radius(s: Setting) -> Quant:
    replicas = max(s.gpu_count // max(s.tp, 1), 1)
    return Quant(
        label="the traffic exposed by a rollout stage",
        steps=[
            f"Fleet holds {replicas} replica(s) serving {s.concurrency} concurrent requests each",
            f"Replacing one replica exposes {1 / replicas * 100:.1f}% of capacity to the new build",
            f"Replacing all at once exposes 100% with no comparison arm remaining",
            "The exposed share is also the share available to detect a fault",
        ],
        value=f"{1 / replicas * 100:.1f}% exposure per replica on a {replicas}-replica fleet",
        interpretation=(
            "Exposure and detectability move together. Too small a stage cannot detect a fault; too "
            "large a stage cannot contain one, and the fleet size bounds how finely this can be tuned."),
    )


def q_timeout_budget(s: Setting) -> Quant:
    hops = 3
    per = s.slo_ms // hops
    return Quant(
        label="how a latency budget must be divided across a call chain",
        steps=[
            f"End-to-end objective is {s.slo_ms} ms across roughly {hops} hops",
            f"If each hop is given the full {s.slo_ms} ms, the chain can take {s.slo_ms * hops} ms",
            f"Dividing the budget gives each hop about {per} ms",
            "A hop whose timeout exceeds its share cannot be preempted by the caller's deadline",
        ],
        value=f"about {per} ms per hop against a {s.slo_ms} ms end-to-end objective",
        interpretation=(
            "Timeouts set independently per hop do not compose into an end-to-end guarantee. The "
            "budget has to be allocated downward from the objective."),
    )


def q_circuit_breaker(s: Setting) -> Quant:
    return Quant(
        label="what a breaker protects and what it does not",
        steps=[
            "A breaker stops sending work to a dependency that is failing",
            "That protects the caller's threads and the dependency's recovery",
            "It does not make the request succeed, and it does not reduce user-visible failure",
            f"At {s.concurrency} concurrency it converts slow failures into fast ones",
        ],
        value="faster failure and protected recovery, not improved success rate",
        interpretation=(
            "A breaker trades availability of the answer for availability of the system. Where the "
            "answer is what the user needs, the breaker is a containment tool rather than a fix."),
    )


def q_graceful_degradation(s: Setting) -> Quant:
    return Quant(
        label="the degraded modes available when full capacity cannot be met",
        steps=[
            f"Full service is {s.concurrency} concurrent requests within {s.slo_ms} ms",
            "Shed the lowest-value traffic class and serve the rest at full quality",
            "Serve all traffic with a smaller model or shorter outputs at reduced quality",
            "Queue and serve late, which meets neither the objective nor the rejection contract",
        ],
        value=f"three degradation choices against a {s.slo_ms} ms objective, only two of them honest",
        interpretation=(
            "Degradation must be chosen in advance, because the default when nothing is chosen is "
            "the third option, which fails the objective silently rather than explicitly."),
    )


def q_dependency_fanout(s: Setting) -> Quant:
    deps = 5
    avail = 0.999
    return Quant(
        label="the availability ceiling imposed by a serial dependency chain",
        steps=[
            f"Suppose {deps} dependencies each available {avail:.1%} of the time",
            f"If all are required, combined availability is {avail}^{deps}",
            f"= {avail ** deps:.4f}, or {avail ** deps * 100:.2f}%",
            f"That is {(1 - avail ** deps) * 100:.2f}% unavailable before the service's own faults",
        ],
        value=f"{avail ** deps * 100:.2f}% ceiling from {deps} required dependencies at {avail:.1%} each",
        interpretation=(
            "Availability multiplies across required dependencies, so a service cannot be more "
            "available than the product of everything it needs. Adding a dependency lowers the "
            "ceiling."),
    )


def q_canary_power(s: Setting) -> Quant:
    replicas = max(s.gpu_count // max(s.tp, 1), 1)
    share = 1 / replicas
    return Quant(
        label="whether a canary stage carries enough traffic to detect the fault it is for",
        steps=[
            f"One replica of {replicas} carries about {share * 100:.1f}% of traffic",
            "A fault affecting one request in a thousand appears once per thousand canary requests",
            f"At {s.concurrency} concurrency, that is a short wait for a common fault and a long one for a rare fault",
            "A canary held for less than that interval cannot observe the fault it exists to catch",
        ],
        value=f"{share * 100:.1f}% of traffic; detection time depends on the fault's rate",
        interpretation=(
            "Canary duration must be derived from the rate of the fault being screened for. A fixed "
            "ten-minute soak is a ritual unless that arithmetic was done."),
    )


def q_rollback_completeness(s: Setting) -> Quant:
    return Quant(
        label="what must be reverted together with a change",
        steps=[
            "The build itself, which is usually the only thing a rollback covers",
            "Configuration tuned while the build was live, such as batch and concurrency limits",
            "Persisted state the build wrote in a new format",
            "Downstream expectations set by the new build's behaviour",
        ],
        value="four categories, of which a standard rollback covers one",
        interpretation=(
            "A rollback that restores only the build can leave the service in a configuration valid "
            "for neither version. The revert set must be recorded when the change is made, not "
            "reconstructed during the incident."),
    )


def q_health_check_depth(s: Setting) -> Quant:
    return Quant(
        label="what a shallow health check actually proves",
        steps=[
            "A process-liveness check proves the process is running",
            "A port check proves the socket is bound",
            f"Neither proves the model loaded, that {gib(s.weight_bytes_per_gpu)} of weights are intact, "
            f"or that a forward pass succeeds",
            "A replica can pass both while returning errors to every request",
        ],
        value=f"liveness and port binding prove neither weights nor inference on {s.gpu_count} devices",
        interpretation=(
            "The check must exercise the path that can fail. For a model server that means a real "
            "forward pass, which is more expensive than a port check and is the only meaningful one."),
    )


def q_retry_storm(s: Setting) -> Quant:
    return Quant(
        label="why recovery is slower than the fault that caused it",
        steps=[
            f"During the fault, clients accumulate pending work and retry budgets",
            "When capacity returns, that backlog arrives at once rather than at the normal rate",
            f"The recovering fleet has cold caches and is slower per request than steady state",
            "Arrival rate is therefore highest exactly when service rate is lowest",
        ],
        value="peak arrivals coincide with minimum capacity at the moment of recovery",
        interpretation=(
            "Recovery needs admission control more than steady state does. A fleet that is restored "
            "and immediately reopened to full traffic will fail again for a different reason."),
    )


register(
    Mechanism(
        key="correlated_failure_domains", topic="reliability",
        title="replicas share enough that independent-failure arithmetic overstates availability",
        concepts=("availability", "failure_domains", "redundancy"),
        symptom="A fleet sized for redundancy lost every replica at once during an incident.",
        chain="Replicas share an image, a configuration rollout, a model artifact and often a rack or power domain, so a fault in any shared element removes all of them together and the redundancy that was purchased does not apply.",
        metric="Number of genuinely independent failure domains, enumerated from what the replicas share rather than from replica count.",
        signature="Observed failures correlate across replicas at a level far above what independence would produce, and the correlation follows a shared element rather than the replica boundary.",
        confounders=(
            "A single common cause outside the deployment, such as an upstream dependency.",
            "A rollout in progress, which correlates replicas deliberately and temporarily.",
            "Placement having drifted so replicas share a rack that the intended layout separated.",
        ),
        fixes=(
            "Enumerate the shared elements and treat their count, not the replica count, as the domain count.",
            "Stagger configuration and artifact rollouts so a bad one cannot reach every replica at once.",
            "Spread placement across power and network domains and verify the actual placement rather than the intent.",
        ),
        rollback="Halt any rollout that has reached more than one domain until the correlation is understood, rather than continuing to the next stage on schedule.",
        options=("staggering rollouts so one change cannot reach every replica", "spreading placement across physical domains"),
        tradeoff="whether the dominant shared element is the software rollout or the physical placement",
        flip="placement is already well spread and the correlation persists, at which point the rollout path rather than the topology is the domain",
        falsifier="failures are distributed across replicas at the rate independence would predict",
        wrong_claim="We run five replicas, so the probability of losing all five simultaneously is negligible.",
        wrong_why="That arithmetic assumes independence, and replicas sharing one image and one rollout path fail together on a single bad artifact, making the effective domain count one rather than five.",
        threshold="Count independent domains from the shared elements and require that count, not the replica count, to meet the availability target.",
        cost="Redundancy purchased against correlated failures buys capacity without buying the availability it was justified by.",
        scaling="Adding replicas raises capacity without raising domain count, so availability plateaus while spend continues to grow.",
        quant=q_correlated_failure,
    ),
    Mechanism(
        key="rollout_stage_sizing", topic="reliability",
        title="rollout exposure and fault detectability are the same dial",
        concepts=("rollout", "canary", "blast_radius"),
        symptom="A staged rollout completed without incident and the fault appeared immediately at full deployment.",
        chain="A stage exposes a share of traffic to the new build, and that same share determines how many requests are available to reveal a fault, so a stage small enough to contain a problem is often too small to observe one.",
        metric="Requests served by the stage compared against the number needed to observe the fault rate being screened for.",
        signature="The stage served fewer requests than the fault's rate requires for a single expected occurrence, so passing the stage carried no information.",
        confounders=(
            "The fault being triggered by traffic the stage did not receive, which is a coverage rather than a volume problem.",
            "The stage running too briefly for a time-dependent fault to develop.",
            "Automatic rollback thresholds being too loose to fire on the observed rate.",
        ),
        fixes=(
            "Size the stage from the fault rate being screened for rather than from a fixed percentage.",
            "Hold the stage long enough for the expected occurrence count to exceed one.",
            "Route a representative traffic sample to the stage rather than whatever load balancing supplies.",
        ),
        rollback="Do not advance a stage that has not served enough requests to observe the fault it screens for, regardless of elapsed time.",
        options=("sizing the stage from the fault rate being screened for", "routing a representative traffic sample to the stage"),
        tradeoff="whether the constraint is volume or traffic representativeness",
        flip="the fleet is too small for any stage to carry enough traffic, at which point representativeness and longer soak are the only available controls",
        falsifier="the stage served far more requests than the screened fault rate requires",
        wrong_claim="The canary ran for thirty minutes with no errors, so the build is safe to roll out.",
        wrong_why="Elapsed time is not evidence; the question is how many requests the stage served against the rate of the fault being screened for, and a small stage can pass on volume alone.",
        threshold="Advance a stage only once its served request count exceeds the reciprocal of the screened fault rate.",
        cost="A rollout that passes every stage without information spends the full rollout window and provides no assurance.",
        scaling="Detectability improves with fleet size while containment worsens, so the two objectives diverge as deployments grow.",
        quant=q_blast_radius,
    ),
    Mechanism(
        key="timeout_budget_composition", topic="reliability",
        title="per-hop timeouts set independently do not compose into an end-to-end guarantee",
        concepts=("timeouts", "deadlines", "slo"),
        symptom="Every service in the chain reports meeting its own timeout while end-to-end latency far exceeds the objective.",
        chain="If each hop is configured with a timeout close to the full end-to-end objective, the chain's worst case is the sum rather than the objective, and no individual hop ever observes a violation.",
        metric="Sum of per-hop timeouts along the chain, compared against the end-to-end objective.",
        signature="The sum exceeds the objective by roughly the hop count, and no hop reports timeout violations.",
        confounders=(
            "Retries within a hop, which multiply that hop's contribution beyond its nominal timeout.",
            "Parallel fan-out, where the contribution is the maximum rather than the sum.",
            "Queue wait outside any hop's timeout accounting.",
        ),
        fixes=(
            "Propagate a deadline from the entry point so each hop inherits the remaining budget.",
            "Allocate the budget explicitly across hops rather than configuring each independently.",
            "Cancel downstream work when the deadline expires rather than letting it complete unused.",
        ),
        rollback="Restore the previous per-hop timeouts if deadline propagation causes premature cancellation of work that was completing within budget.",
        options=("propagating a deadline from the entry point", "allocating the budget explicitly across hops"),
        tradeoff="whether every hop in the chain can be modified to honour a propagated deadline",
        flip="a third-party hop cannot accept a deadline, at which point explicit static allocation is the only enforceable scheme",
        falsifier="the sum of per-hop timeouts is already within the end-to-end objective",
        wrong_claim="Each service has a timeout well within our latency objective, so the chain is bounded by it.",
        wrong_why="Timeouts along a serial chain add rather than bound one another, so a chain of hops each within the objective can take a multiple of it.",
        threshold="Require the sum of per-hop timeouts along the critical path to sit within the end-to-end objective.",
        cost="Work performed after the caller has abandoned the request consumes capacity and produces nothing.",
        scaling="The discrepancy grows linearly with chain length, so it worsens as architectures decompose into more services.",
        quant=q_timeout_budget,
    ),
    Mechanism(
        key="breaker_scope", topic="reliability",
        title="a circuit breaker protects the system and does not improve the user's outcome",
        concepts=("circuit_breaker", "containment", "availability"),
        symptom="A breaker was added after an incident and the next incident produced the same user-visible failure rate, faster.",
        chain="Opening a breaker stops work reaching a failing dependency, which frees the caller's resources and lets the dependency recover, but the request that needed the dependency still cannot be answered, so user-visible success is unchanged.",
        metric="Caller resource exhaustion and dependency recovery time, measured separately from request success rate.",
        signature="Caller saturation disappears and dependency recovery shortens while the failed-request count is unchanged.",
        confounders=(
            "A fallback path being added alongside the breaker, which does change success rate.",
            "Retries being reduced at the same time, which reduces load independently.",
            "The dependency recovering faster for unrelated reasons.",
        ),
        fixes=(
            "State the breaker's purpose as containment and measure it on containment metrics.",
            "Add an explicit fallback if the user outcome must improve, since the breaker alone will not do it.",
            "Tune the half-open probe rate so recovery is detected without re-saturating the dependency.",
        ),
        rollback="Close the breaker permanently if it opens on transient conditions that would have succeeded, since a breaker that trips too easily reduces availability rather than protecting it.",
        options=("measuring the breaker on containment metrics", "adding an explicit fallback path"),
        tradeoff="whether the objective is protecting the system or answering the request",
        flip="the request genuinely has no acceptable fallback, at which point containment is the only available goal and success rate is not the measure",
        falsifier="user-visible success rate improves after the breaker is added with nothing else changed",
        wrong_claim="We added a circuit breaker, so the service is now resilient to that dependency failing.",
        wrong_why="The breaker prevents the failure from consuming the caller's resources; it does not produce an answer, so requests requiring that dependency still fail.",
        threshold="Judge a breaker on caller saturation and dependency recovery, never on request success rate.",
        cost="Believing a breaker restores availability defers building the fallback that actually would.",
        scaling="Containment value grows with fan-in to the dependency, so breakers matter more as more callers share it.",
        quant=q_circuit_breaker,
    ),
    Mechanism(
        key="degradation_default", topic="reliability",
        title="if no degraded mode is chosen in advance, the default is the worst one",
        concepts=("graceful_degradation", "load_shedding", "slo"),
        symptom="Under overload the service neither rejects nor degrades; it serves everything late and breaches its objective for all traffic.",
        chain="Degradation requires a policy choice about what to give up, and in the absence of one the system queues everything, which spreads the shortfall across all traffic instead of concentrating it where it costs least.",
        metric="Share of traffic meeting the objective under overload, compared across the available degradation policies.",
        signature="Under overload every traffic class degrades together rather than a chosen class degrading first.",
        confounders=(
            "A shedding policy existing but with thresholds set too high to activate.",
            "Client retries converting shed load back into offered load.",
            "One traffic class dominating volume, so shedding it looks like shedding everything.",
        ),
        fixes=(
            "Choose and implement a shedding policy by traffic class before the next overload.",
            "Define a reduced-quality mode, such as shorter outputs or a smaller model, as an explicit alternative to shedding.",
            "Set the thresholds from the measured objective rather than from a round utilisation number.",
        ),
        rollback="Return to uniform queueing only if shedding is found to remove traffic that was more valuable than what it preserved, and record that finding rather than reverting silently.",
        options=("shedding a chosen traffic class", "serving all traffic in a reduced-quality mode"),
        tradeoff="whether the traffic classes differ enough in value to make shedding preferable to degrading",
        flip="all traffic is equally valuable, at which point uniform quality reduction is fairer than shedding a class",
        falsifier="one class already degrades before the others under overload",
        wrong_claim="The service degrades gracefully because it queues rather than dropping requests.",
        wrong_why="Queueing everything is the absence of a degradation policy; it converts an overload into a universal objective breach rather than concentrating the loss where it was chosen to fall.",
        threshold="Require a named degraded mode with activation thresholds before a service is considered to have overload handling.",
        cost="A universal breach damages every traffic class including the one that could have been protected at no additional cost.",
        scaling="The cost of having no policy grows with traffic heterogeneity, since more value is available to protect and none of it is.",
        quant=q_graceful_degradation,
    ),
    Mechanism(
        key="dependency_availability_ceiling", topic="reliability",
        title="required dependencies multiply, so each one lowers the achievable availability ceiling",
        concepts=("availability", "dependencies", "architecture"),
        symptom="A service with a high availability target cannot reach it despite the service itself rarely failing.",
        chain="If every dependency is required for a request to succeed, availability is the product of the dependencies' availabilities, so the ceiling falls with each addition regardless of how reliable the service's own code is.",
        metric="Product of the availabilities of all required dependencies, computed before the service's own contribution.",
        signature="Observed availability sits close to the dependency product, and the service's own fault rate accounts for only a small remainder.",
        confounders=(
            "Dependency failures being correlated, which makes the product pessimistic.",
            "Some dependencies being optional in practice, which raises the real ceiling.",
            "Caching masking a dependency's unavailability for part of the traffic.",
        ),
        fixes=(
            "Make dependencies optional with a defined degraded behaviour wherever the product permits.",
            "Cache dependency results so transient unavailability does not fail the request.",
            "Remove dependencies that contribute less value than the availability they cost.",
        ),
        rollback="Restore a dependency to required status if the degraded behaviour produces incorrect results, since availability is not worth correctness.",
        options=("making a dependency optional with defined degraded behaviour", "caching results so transient unavailability is tolerated"),
        tradeoff="whether the request can be answered acceptably without the dependency",
        flip="the dependency's data is required for correctness rather than enrichment, at which point it cannot be made optional and only its own availability matters",
        falsifier="observed availability is far above the dependency product",
        wrong_claim="Our service code is highly reliable, so we can commit to a high availability target.",
        wrong_why="The target is bounded by the product of everything the request requires, so a reliable service behind several required dependencies cannot exceed their combined availability.",
        threshold="Set the availability target no higher than the product of required dependency availabilities.",
        cost="Committing to a target the architecture cannot reach guarantees breach and the remediation work that follows it.",
        scaling="The ceiling falls geometrically with dependency count, so decomposition lowers achievable availability unless dependencies are made optional.",
        quant=q_dependency_fanout,
    ),
    Mechanism(
        key="canary_detection_power", topic="reliability",
        title="a canary must be sized from the fault rate it screens for, not from the clock",
        concepts=("canary", "statistical_power", "rollout"),
        symptom="Canary stages pass consistently and faults are found by users after full rollout.",
        chain="A canary can only observe a fault if it serves enough requests for at least one occurrence to be expected, so a stage sized by elapsed time rather than by the screened fault rate passes without having tested anything.",
        metric="Expected occurrences during the stage, computed as served requests times the screened fault rate.",
        signature="The expected occurrence count during the stage is well below one, so passing was the likely outcome whether or not the fault existed.",
        confounders=(
            "The canary receiving unrepresentative traffic, which is a coverage rather than a power problem.",
            "Automatic analysis thresholds being too loose to fire on the observed count.",
            "The fault being deterministic on a rare input rather than probabilistic.",
        ),
        fixes=(
            "Compute expected occurrences before the stage and extend it until the count exceeds one.",
            "Raise the traffic share rather than the duration where fleet size permits.",
            "Screen for the fault with a targeted probe rather than relying on organic traffic.",
        ),
        rollback="Do not treat a stage as passed when its expected occurrence count was below one; record it as uninformative rather than as evidence.",
        options=("extending the stage until expected occurrences exceed one", "screening with a targeted probe instead of organic traffic"),
        tradeoff="whether the fault can be provoked deliberately or only observed in organic traffic",
        flip="the fault depends on rare organic inputs that cannot be synthesised, at which point only volume and duration can raise the power",
        falsifier="expected occurrences during the stage were comfortably above one",
        wrong_claim="The canary saw no errors, so the build does not contain this class of fault.",
        wrong_why="With an expected occurrence count below one, seeing no errors is the most likely outcome even when the fault is present, so the observation carries almost no information.",
        threshold="Require expected occurrences above one before a canary stage counts as evidence.",
        cost="A rollout process that produces no information still consumes the full rollout window and the engineering attention around it.",
        scaling="Required stage size grows as the reciprocal of the fault rate, so rare-but-severe faults are the hardest to screen and the most damaging to miss.",
        quant=q_canary_power,
    ),
    Mechanism(
        key="incomplete_rollback", topic="reliability",
        title="reverting the build leaves configuration and state that were valid only under it",
        concepts=("rollback", "configuration", "incident_response"),
        symptom="A rollback completed successfully and the service remained degraded in a new way.",
        chain="A change is usually accompanied by configuration tuned for it and by state written in its format, and a rollback that restores only the build leaves the old build running against settings and data it was never validated with.",
        metric="The recorded revert set for the change, compared against what the rollback actually restored.",
        signature="The post-rollback failure differs from the pre-rollback one and corresponds to a setting or data format introduced with the change.",
        confounders=(
            "An unrelated change deployed in the same window.",
            "The original fault persisting because the build was not its cause.",
            "Caches holding data written in the new format after the revert.",
        ),
        fixes=(
            "Record the full revert set when the change is made rather than reconstructing it during an incident.",
            "Make forward-compatible state changes so the previous build can read what the new one wrote.",
            "Revert configuration and build together as one unit.",
        ),
        rollback="Roll forward to a fixed build rather than continuing to revert if the revert set cannot be established during the incident, since a partial revert compounds the failure.",
        options=("recording the full revert set at change time", "making state changes forward-compatible"),
        tradeoff="whether the state written by the new build can be read by the old one",
        flip="the state format cannot be made backward-readable, at which point rolling forward is safer than rolling back",
        falsifier="the post-rollback failure is identical to the pre-rollback one",
        wrong_claim="We rolled back to the previous version, so the service is in a known-good state.",
        wrong_why="The previous version is now running against configuration tuned for the new one and state written in its format, which is a combination that was never tested.",
        threshold="Require a recorded revert set covering build, configuration and state before a change is deployed.",
        cost="A partial revert extends the incident and introduces a second failure mode on top of the first.",
        scaling="The revert set grows with the number of coupled subsystems, so rollback becomes less reliable as systems integrate more tightly.",
        quant=q_rollback_completeness,
    ),
    Mechanism(
        key="shallow_health_check", topic="reliability",
        title="a health check that does not run inference cannot detect a model server that is broken",
        concepts=("health_checks", "readiness", "serving"),
        symptom="A replica passes health checks and returns errors or degraded output to every request it receives.",
        chain="Liveness and port checks prove the process runs and the socket is bound, and neither exercises weight loading, device state or the forward pass, so a replica whose model failed to load correctly remains in rotation.",
        metric="Whether the readiness check performs a real forward pass and validates its output.",
        signature="The replica passes every configured check while its request error rate is far above its peers'.",
        confounders=(
            "A dependency failure affecting only real requests, which a forward-pass check would also pass.",
            "Load balancer caching a stale health state.",
            "The check running against a different process than the one serving.",
        ),
        fixes=(
            "Make readiness perform a real forward pass and compare against a fixed expected output.",
            "Include artifact hash verification in the readiness path.",
            "Remove a replica from rotation on elevated error rate rather than waiting for a health check to fail.",
        ),
        rollback="Loosen the readiness check if it removes healthy replicas under load, since an over-strict check converts a capacity problem into an outage.",
        options=("making readiness perform a real forward pass", "removing replicas on elevated error rate"),
        tradeoff="whether the failure is detectable from inside the replica or only from its request outcomes",
        flip="the check itself becomes expensive enough to affect serving capacity, at which point outcome-based ejection is the cheaper signal",
        falsifier="the failing replica also fails its configured health check",
        wrong_claim="All replicas report healthy, so the errors must come from somewhere else in the stack.",
        wrong_why="The checks in use verify process liveness and port binding, neither of which exercises the model path where the failure lives, so a healthy report is expected regardless.",
        threshold="Require readiness to exercise the inference path before a replica is allowed to receive traffic.",
        cost="Traffic routed to a broken replica fails at the share of the fleet that replica represents, for as long as it stays in rotation.",
        scaling="The exposure is proportional to the failing replica's traffic share, so small fleets suffer more per broken replica.",
        quant=q_health_check_depth,
    ),
    Mechanism(
        key="recovery_thundering_herd", topic="reliability",
        title="recovery presents peak arrivals to a fleet that is at its slowest",
        concepts=("recovery", "admission_control", "cold_cache"),
        symptom="Service is restored, immediately fails again, and the second failure is harder to explain than the first.",
        chain="During the outage clients accumulate pending work and retry budgets, so when capacity returns the backlog arrives at once, while the recovering fleet has empty caches and is slower per request than it will be in steady state.",
        metric="Arrival rate and per-request service time during the recovery window, compared against steady-state values.",
        signature="Arrival rate peaks and service time is at its worst simultaneously, at the moment capacity is restored.",
        confounders=(
            "The original fault not being fixed, so the second failure is a continuation.",
            "A deployment occurring during recovery, which adds its own instability.",
            "Client retry behaviour differing from what was assumed, which changes the backlog shape.",
        ),
        fixes=(
            "Reopen capacity gradually rather than restoring full traffic at once.",
            "Admit traffic by priority during recovery so the backlog does not displace new requests.",
            "Warm caches with synthetic traffic before accepting organic load.",
        ),
        rollback="Return to gradual reopening if a full restore fails again, and record the recovery admission policy so it is not rediscovered during the next incident.",
        options=("reopening capacity gradually", "warming caches before accepting organic load"),
        tradeoff="whether the fleet's slowness at recovery comes from cold caches or from the backlog volume",
        flip="the backlog is large enough that no warm-up helps before it arrives, at which point admission control is the only effective lever",
        falsifier="arrival rate at recovery is close to steady-state levels",
        wrong_claim="Capacity is restored, so we can return the service to full traffic.",
        wrong_why="Restored capacity is cold and slower than steady state, while the arriving load includes the whole accumulated backlog, so the fleet faces its highest demand at its lowest capability.",
        threshold="Reopen at a rate the recovering fleet's measured cold service time can absorb, not at full traffic.",
        cost="A second failure during recovery extends the incident and consumes the credibility of the first recovery.",
        scaling="Backlog size grows with outage duration, so longer outages make the recovery itself progressively more dangerous.",
        quant=q_retry_storm,
    ),
)
