"""Storage, checkpointing and artifact-path mechanisms (topic: storage)."""
from __future__ import annotations

from core import Mechanism, Quant, Setting, fmt_int, gib, register


def q_page_cache(s: Setting) -> Quant:
    return Quant(
        label="how much of a repeated read benchmark never reaches the storage device",
        steps=[
            f"The artifact being read is {gib(s.weight_bytes)}",
            "The operating system retains recently read pages in host memory",
            "A second pass over the same file is served from memory rather than from the device",
            "Reported throughput then measures memory bandwidth, not storage bandwidth",
        ],
        value=f"a second pass over {gib(s.weight_bytes)} can be served entirely from host memory",
        interpretation=(
            "Cache state must be stated with any storage figure. Without it the number describes an "
            "unknown mixture of two subsystems whose speeds differ by an order of magnitude."),
    )


def q_load_parallel(s: Setting) -> Quant:
    per = s.weight_bytes_per_gpu
    return Quant(
        label="the transfer each rank performs during a parallel weight load",
        steps=[
            f"Total artifact {gib(s.weight_bytes)}, sharded across TP{s.tp}",
            f"Each rank needs {gib(per)} of it",
            f"If all {s.tp} ranks read from one source, that source serves "
            f"{gib(s.weight_bytes)} concurrently",
            "Per-rank time is then set by the source's aggregate capacity divided by rank count",
        ],
        value=f"{gib(per)} per rank, {gib(s.weight_bytes)} demanded from the source at once",
        interpretation=(
            "Parallel loading converts a bandwidth problem into a fan-out problem. Adding readers "
            "helps only while the source can serve them, and beyond that point it adds contention."),
    )


def q_checkpoint_size(s: Setting) -> Quant:
    weights = s.weight_bytes
    optim = weights * 2
    return Quant(
        label="the difference between an inference artifact and a training checkpoint",
        steps=[
            f"Weights alone: {gib(weights)}",
            f"Optimiser state for a common two-moment optimiser adds roughly twice that: {gib(optim)}",
            f"Total checkpoint: {gib(weights + optim)}",
            "Only the weight portion is needed to serve; the rest exists to resume training",
        ],
        value=f"{gib(weights)} to serve against roughly {gib(weights + optim)} to resume",
        interpretation=(
            "Sizing storage and transfer budgets from the served artifact underestimates training "
            "checkpoints by roughly a factor of three, which is where checkpoint stalls originate."),
    )


def q_write_stall(s: Setting) -> Quant:
    total = s.weight_bytes * 3
    return Quant(
        label="the time a synchronous checkpoint write holds the accelerators idle",
        steps=[
            f"Checkpoint payload is roughly {gib(total)} including optimiser state",
            "A synchronous write blocks the training step until it completes",
            f"At an achieved 1 GiB/s that is {total / (1024 ** 3):.0f} seconds of idle accelerators",
            f"Across {s.gpu_count} devices that is "
            f"{total / (1024 ** 3) * s.gpu_count / 3600:.2f} device-hours per checkpoint",
        ],
        value=f"{total / (1024 ** 3) * s.gpu_count / 3600:.2f} device-hours idle per synchronous checkpoint",
        interpretation=(
            "Checkpoint frequency is a trade between this cost and the work lost on failure. Both "
            "terms are measurable, so the interval should be derived rather than chosen by habit."),
    )


def q_object_latency(s: Setting) -> Quant:
    parts = max(int(s.weight_bytes / (64 * 1024 * 1024)), 1)
    return Quant(
        label="the request count a large artifact becomes on an object store",
        steps=[
            f"Artifact size {gib(s.weight_bytes)}",
            f"At a 64 MiB part size that is {fmt_int(parts)} separate object requests",
            "Each request carries a round trip whose latency is independent of part size",
            f"Serialised, {fmt_int(parts)} round trips dominate; parallelised, they contend for bandwidth",
        ],
        value=f"{fmt_int(parts)} object requests for one {gib(s.weight_bytes)} artifact",
        interpretation=(
            "Object stores price latency per request rather than per byte. Concurrency and part size "
            "are therefore the two parameters that decide load time, and neither is the store's "
            "advertised throughput."),
    )


def q_format_conversion(s: Setting) -> Quant:
    return Quant(
        label="the cost of converting an artifact format at load time rather than at build time",
        steps=[
            f"A conversion reads {gib(s.weight_bytes)} and writes an equivalent volume",
            f"Done per process start, it is paid on every replica and every restart",
            f"Across {s.gpu_count // max(s.tp, 1)} replicas that is "
            f"{gib(s.weight_bytes * (s.gpu_count // max(s.tp, 1)))} of redundant work",
            "Done once at build time, it is paid once per artifact version",
        ],
        value=f"{gib(s.weight_bytes)} converted per replica start versus once per build",
        interpretation=(
            "Load-time conversion is invisible in a single-replica test and multiplies across the "
            "fleet. It is one of the few costs that grows with replica count rather than with load."),
    )


def q_tiering(s: Setting) -> Quant:
    return Quant(
        label="what tier an artifact must sit in to meet a given start-up target",
        steps=[
            f"Artifact {gib(s.weight_bytes)} must move before the replica serves",
            "A remote object tier delivers it over the network at whatever the path allows",
            "A local disk tier delivers it at device speed with no network involved",
            "Host page cache delivers it at memory speed and holds only what fits alongside everything else",
        ],
        value=f"{gib(s.weight_bytes)} must be resident in a tier fast enough for the start-up target",
        interpretation=(
            "Start-up time is chosen by tier placement rather than by tuning. The decision is where "
            "the bytes live before they are needed, not how quickly they are fetched when they are."),
    )


def q_read_amplification(s: Setting) -> Quant:
    useful = s.weight_bytes_per_gpu
    read = s.weight_bytes
    return Quant(
        label="the read amplification when every rank reads the whole artifact",
        steps=[
            f"Each rank needs only its shard: {gib(useful)}",
            f"If each reads the full file and discards the rest, it reads {gib(read)}",
            f"Amplification factor = {read / max(useful, 1):.1f}x per rank",
            f"Across TP{s.tp} ranks the source serves {gib(read * s.tp)} to deliver {gib(read)} of useful data",
        ],
        value=f"{read / max(useful, 1):.1f}x read amplification per rank",
        interpretation=(
            "Amplification is invisible in per-rank timings and obvious at the source. It is the usual "
            "reason a load that is fast on one rank is slow when all ranks start together."),
    )


def q_storage_share(s: Setting) -> Quant:
    return Quant(
        label="the share of an operation that a storage optimisation can actually address",
        steps=[
            f"Loading {gib(s.weight_bytes)} is a one-time cost per process",
            f"Serving then runs for hours against a {s.slo_ms} ms per-request objective",
            "Storage appears on the critical path only at start and at checkpoint boundaries",
            "So the upper bound on end-to-end benefit is the share of wall time those events occupy",
        ],
        value="the benefit is bounded by the share of wall time spent loading or checkpointing",
        interpretation=(
            "Compute that bound before benchmarking the storage path. If it is small, a large relative "
            "storage improvement still produces a negligible end-to-end effect."),
    )


def q_integrity(s: Setting) -> Quant:
    return Quant(
        label="why a silently corrupted artifact is worse than a failed load",
        steps=[
            f"An artifact of {gib(s.weight_bytes)} is transferred across a network and a filesystem",
            "A failed transfer raises and the replica does not start",
            "A partially corrupted transfer may still parse and load successfully",
            "The replica then serves degraded output with no error anywhere in the system",
        ],
        value=f"{gib(s.weight_bytes)} transferred with no integrity check produces silent degradation",
        interpretation=(
            "A content hash verified after transfer converts a silent quality fault into a loud "
            "start-up failure. That trade is almost always correct for a serving artifact."),
    )


register(
    Mechanism(
        key="page_cache_confound", topic="storage",
        title="repeated-read storage benchmarks measure host memory rather than the device",
        concepts=("benchmarking", "page_cache", "storage"),
        symptom="Storage throughput measured in a loop is far above the device's rated capability and does not reproduce in production.",
        chain="The operating system retains recently read pages in host memory, so every pass after the first is served from memory, and the benchmark reports memory bandwidth under a storage label.",
        metric="Device-level read counters compared against the bytes the benchmark believes it read.",
        signature="Device read counters are near zero on the repeated passes while the benchmark continues to report high throughput.",
        confounders=(
            "The storage device's own cache, which cannot be dropped from the host and produces a smaller version of the same effect.",
            "A working set that genuinely fits in memory, in which case the production path also benefits.",
            "Readahead prefetching, which reduces apparent latency without eliminating device reads.",
        ),
        fixes=(
            "Drop caches between passes and report the cache condition with every figure.",
            "Use a working set larger than host memory where dropping caches is not permitted.",
            "Corroborate every storage figure against device-level read counters.",
        ),
        rollback="Discard any published storage figure that does not state its cache condition, rather than reinterpreting it after the fact.",
        options=("dropping caches between passes and reporting the condition", "using a working set larger than host memory"),
        tradeoff="whether the environment permits dropping caches or only permits enlarging the working set",
        flip="the container cannot drop caches and memory is large enough that no practical working set exceeds it, at which point only device counters can settle the question",
        falsifier="device read counters account for the full volume on every pass",
        wrong_claim="We measured 8 GB/s sustained read, so the storage path is not the bottleneck.",
        wrong_why="A figure above the device's rated capability indicates the reads were served from host memory, so it describes a path production will not take.",
        threshold="Require a stated cache condition and corroborating device counters before any storage figure is used in a decision.",
        cost="Provisioning decisions made on cached benchmark numbers buy the wrong tier and are discovered only at the next cold start.",
        scaling="The effect is strongest for artifacts smaller than host memory, so it misleads most on exactly the mid-sized models that are easiest to test.",
        quant=q_page_cache,
    ),
    Mechanism(
        key="parallel_load_fanout", topic="storage",
        title="parallel weight loading turns a bandwidth question into a fan-out question",
        concepts=("weight_loading", "fan_out", "startup"),
        symptom="Loading is fast when one rank starts and slow when the whole group starts together.",
        chain="Every rank fetches its shard from the same source at the same moment, so the source must serve the entire artifact concurrently, and per-rank time becomes the source's aggregate capacity divided by the number of readers.",
        metric="Source-side served bytes and concurrent reader count during start-up, alongside per-rank load time.",
        signature="Per-rank load time scales with reader count while each rank's own transferred volume is unchanged.",
        confounders=(
            "Network path contention between the readers, which slows them for a reason outside the source.",
            "Cold page cache on first start, which affects the single-rank case as well.",
            "Rate limiting at the source, which caps aggregate throughput independently of capacity.",
        ),
        fixes=(
            "Stagger rank start so the source's concurrency stays within what it can serve.",
            "Replicate the artifact to a per-host or per-rack cache so readers fan out across sources.",
            "Pre-stage shards to the nodes that will read them, removing the fetch from the start path.",
        ),
        rollback="Remove staggering if it lengthens total start-up beyond what the concurrency cost was, since staggering trades parallelism for source relief.",
        options=("staggering rank start to bound source concurrency", "replicating the artifact to per-host caches"),
        tradeoff="whether the source can serve the full reader count at useful per-reader bandwidth",
        flip="the artifact grows large enough that replication storage cost exceeds the start-up time it saves, at which point staggering is the cheaper control",
        falsifier="per-rank load time is unchanged as reader count rises",
        wrong_claim="Loading takes two minutes in our single-node test, so a full cluster start will also take two minutes.",
        wrong_why="A single reader measures the source at concurrency one, and the cluster case presents the source with every rank at once, which is a different and usually much slower operating point.",
        threshold="Bound concurrent readers per source to the count at which per-reader bandwidth stays acceptable.",
        cost="Every rank waiting on a saturated source holds its devices idle for the whole of the extended load.",
        scaling="Source demand grows linearly with rank count, so the problem appears abruptly at the scale where the source saturates.",
        quant=q_load_parallel,
    ),
    Mechanism(
        key="checkpoint_vs_artifact_size", topic="storage",
        title="a training checkpoint is several times the size of the artifact that serves",
        concepts=("checkpointing", "optimizer_state", "capacity_planning"),
        symptom="Storage and transfer budgets sized from the model size are exhausted by training checkpoints.",
        chain="A checkpoint must contain optimiser state as well as weights so training can resume, and common optimisers hold two additional tensors per parameter, so the checkpoint is roughly three times the served artifact.",
        metric="Checkpoint bytes on disk compared against the served artifact bytes for the same model.",
        signature="The ratio is close to three for a two-moment optimiser and close to one for an inference-only export, which identifies which artifact is being measured.",
        confounders=(
            "Mixed-precision training holding a separate master copy, which changes the ratio again.",
            "Sharded checkpoints spreading the same total across files, which hides the aggregate.",
            "Compression applied to one artifact and not the other.",
        ),
        fixes=(
            "Size storage and transfer budgets from the checkpoint rather than from the model size.",
            "Export an inference-only artifact separately so serving does not carry optimiser state.",
            "Shard the checkpoint across the write path so no single target absorbs the whole volume.",
        ),
        rollback="If reducing checkpoint retention to fit a budget, record the resulting recovery-point exposure explicitly rather than treating retention as a storage parameter.",
        options=("exporting an inference-only artifact separately", "sharding the checkpoint across the write path"),
        tradeoff="whether the volume problem is on the serving path or the training path",
        flip="the training job's write path becomes the constraint rather than serving storage, at which point separating the export does not help and sharding is required",
        falsifier="checkpoint size and served artifact size are close to equal",
        wrong_claim="The model is 70B parameters at bf16, so a checkpoint is about 140 GB.",
        wrong_why="That figure covers weights only; the optimiser state that makes the checkpoint resumable is typically twice again, so the real requirement is roughly three times the quoted number.",
        threshold="Budget checkpoint storage at the measured weights-plus-optimiser ratio for the optimiser actually in use.",
        cost="Under-budgeted checkpoint storage stalls training runs whose accelerator time is already committed.",
        scaling="The ratio is fixed by the optimiser, so the absolute shortfall grows linearly with model size.",
        quant=q_checkpoint_size,
    ),
    Mechanism(
        key="synchronous_checkpoint_stall", topic="storage",
        title="a synchronous checkpoint holds every accelerator idle for the whole write",
        concepts=("checkpointing", "utilisation", "training"),
        symptom="Training throughput shows regular deep dips whose period matches the checkpoint interval.",
        chain="A synchronous checkpoint blocks the step until the write completes, so every device in the job waits for a transfer whose duration is set by the write path rather than by the compute.",
        metric="Device idle time per checkpoint event, multiplied by device count and checkpoint frequency.",
        signature="The dips align exactly with checkpoint boundaries and their depth tracks write-path throughput rather than batch size.",
        confounders=(
            "Evaluation passes scheduled at the same interval, which produce similar dips for a different reason.",
            "Learning-rate schedule boundaries coinciding with checkpoints.",
            "Filesystem contention from another job, which lengthens writes unpredictably.",
        ),
        fixes=(
            "Move the checkpoint write off the critical path by staging to host memory and flushing asynchronously.",
            "Shard the write across ranks so each writes a slice rather than one rank writing everything.",
            "Derive the checkpoint interval from measured write cost and measured failure rate rather than choosing it by habit.",
        ),
        rollback="Return to synchronous writes if asynchronous staging produces checkpoints that are incomplete after a crash, since a fast unusable checkpoint is worse than a slow usable one.",
        options=("staging the write to host memory and flushing asynchronously", "sharding the write across ranks"),
        tradeoff="whether an asynchronously flushed checkpoint can be guaranteed complete and consistent",
        flip="crash consistency cannot be guaranteed for the staged path, at which point sharding the synchronous write is the only safe acceleration",
        falsifier="the throughput dips do not align with checkpoint boundaries",
        wrong_claim="Checkpointing every ten minutes is cheap insurance against failure.",
        wrong_why="The insurance premium is the whole job's devices idled for the write duration each time, and whether it is cheap depends on write cost against failure rate, neither of which was measured.",
        threshold="Set the checkpoint interval where marginal write cost equals the marginal expected work lost to failure.",
        cost="Idle device-hours during checkpoint writes are billed identically to training device-hours and produce nothing.",
        scaling="Write volume grows with model size while failure rate grows with job size, so both terms of the interval calculation move as jobs scale.",
        quant=q_write_stall,
    ),
    Mechanism(
        key="object_store_request_count", topic="storage",
        title="an object store prices latency per request, so part size and concurrency decide load time",
        concepts=("object_storage", "latency", "startup"),
        symptom="An object store rated for high aggregate throughput delivers a large artifact slowly during replica start.",
        chain="A large artifact is fetched as many separate ranged requests, each carrying a round trip whose latency does not shrink with part size, so serialised fetching is dominated by round trips rather than by the store's throughput rating.",
        metric="Request count, per-request latency and achieved concurrency during the fetch, alongside total bytes.",
        signature="Achieved throughput equals request concurrency times part size divided by per-request latency, and it tracks concurrency rather than the store's rating.",
        confounders=(
            "Throttling by the store at high request rates, which caps concurrency for a different reason.",
            "Connection setup cost when connections are not reused, which adds a second per-request term.",
            "Network path bandwidth becoming the limit once concurrency is high enough.",
        ),
        fixes=(
            "Raise fetch concurrency until either the store throttles or the network path saturates.",
            "Increase part size so fewer round trips carry the same volume.",
            "Cache the artifact closer to the readers so the fetch happens once rather than per replica.",
        ),
        rollback="Reduce concurrency if the store begins throttling or returning errors, since a throttled fetch is slower than a moderately concurrent one.",
        options=("raising fetch concurrency", "increasing the part size"),
        tradeoff="whether the limit reached first is request latency, store throttling or path bandwidth",
        flip="the store starts throttling before the path saturates, at which point larger parts rather than more requests is the remaining lever",
        falsifier="achieved throughput matches the store's rating at low concurrency",
        wrong_claim="The bucket is rated for tens of gigabytes per second, so fetching the model will not be slow.",
        wrong_why="That rating describes aggregate capacity across many concurrent clients, while a single sequential fetch is limited by round-trip latency times request count, which the rating does not bound.",
        threshold="Size concurrency and part size so the fetch is bounded by path bandwidth rather than by request round trips.",
        cost="Replicas waiting on a serialised fetch hold their devices idle for the entire start-up window.",
        scaling="Request count grows linearly with artifact size at fixed part size, so larger models degrade more than proportionally under a serial fetch.",
        quant=q_object_latency,
    ),
    Mechanism(
        key="load_time_conversion", topic="storage",
        title="format conversion at load time is paid per replica instead of once per build",
        concepts=("artifacts", "startup", "build_pipeline"),
        symptom="Start-up is slower than the artifact size and storage bandwidth together can explain, on every replica.",
        chain="If the serving runtime converts the stored format before use, the conversion reads and rewrites the whole artifact on each process start, so a cost that belongs to the build pipeline is charged to every replica and every restart.",
        metric="Time between file read completing and the first forward pass, isolated from transfer time.",
        signature="A CPU-bound interval appears after the read completes and before serving begins, and its duration is independent of storage speed.",
        confounders=(
            "Graph capture and autotuning, which also occupy a CPU-bound interval after load.",
            "Weight sharding performed at load time, which is a similar but separately fixable cost.",
            "Decompression, which resembles conversion and is fixed the same way.",
        ),
        fixes=(
            "Convert once in the build pipeline and publish the runtime's native format as the artifact.",
            "Cache the converted artifact on the node after the first conversion.",
            "Choose a storage format the runtime consumes directly, even at some size cost.",
        ),
        rollback="Return to load-time conversion if the pre-converted artifact ties the deployment to a runtime version that cannot be upgraded independently.",
        options=("caching the converted artifact on the node", "converting once in the build pipeline"),
        tradeoff="whether the converted format can be pinned without coupling the artifact to a runtime version",
        flip="the runtime version changes frequently enough that pre-converted artifacts must be rebuilt each time, at which point node-side caching is the more flexible choice",
        falsifier="the interval between read completion and first forward pass is negligible",
        wrong_claim="Start-up is slow because storage is slow, so we should move to a faster tier.",
        wrong_why="The interval after the read completes is CPU-bound conversion, which a faster storage tier does not shorten at all.",
        threshold="Require start-up to be accounted for by transfer time plus a stated capture interval; investigate any unexplained remainder.",
        cost="A conversion repeated on every replica multiplies a one-time build cost by the fleet size and by the restart rate.",
        scaling="Conversion time grows with parameter count, so the per-replica penalty grows with model size while the build-time alternative does not.",
        quant=q_format_conversion,
    ),
    Mechanism(
        key="artifact_tier_placement", topic="storage",
        title="start-up time is decided by which tier the artifact already sits in",
        concepts=("caching", "tiering", "startup"),
        symptom="Start-up time varies by an order of magnitude across nodes running identical configuration.",
        chain="The artifact is fetched from whichever tier currently holds it, and nodes differ in whether it is in page cache, on local disk or only in remote object storage, so identical configuration produces very different start times.",
        metric="Start-up time grouped by which tier served the artifact on that node.",
        signature="Start-up time clusters into distinct bands that correspond exactly to tier, rather than varying continuously.",
        confounders=(
            "Node hardware differences, which also produce banded start times.",
            "Concurrent starts on the same node competing for the same local tier.",
            "Eviction from the local tier between runs, which moves a node between bands.",
        ),
        fixes=(
            "Pre-stage the artifact to the local tier on candidate nodes before scheduling work to them.",
            "Pin the artifact in the local tier with an explicit retention policy rather than relying on eviction order.",
            "Constrain placement to nodes that already hold the artifact, where start-up time matters more than packing.",
        ),
        rollback="Remove the placement constraint if it fragments the schedulable pool more than the start-up saving is worth.",
        options=("pre-staging the artifact to candidate nodes", "constraining placement to nodes that already hold it"),
        tradeoff="whether pre-staging storage is cheaper than the scheduling flexibility a placement constraint costs",
        flip="the artifact set grows too large to pre-stage everywhere, at which point placement constraints become the only workable control",
        falsifier="start-up time varies continuously rather than clustering by tier",
        wrong_claim="All the nodes have the same configuration, so start-up time should be consistent.",
        wrong_why="Configuration does not determine where the bytes currently live, and tier residency differs per node as a result of past scheduling rather than of configuration.",
        threshold="Treat start-up time bands that correspond to tiers as a placement problem rather than a tuning problem.",
        cost="Nodes fetching from the slowest tier hold their devices idle for the whole of an avoidable transfer.",
        scaling="Local tier capacity is fixed while the artifact set grows, so residency falls and slow starts become more common over time.",
        quant=q_tiering,
    ),
    Mechanism(
        key="shard_read_amplification", topic="storage",
        title="ranks reading the whole artifact to keep one shard multiply source load by the parallel degree",
        concepts=("weight_loading", "sharding", "read_amplification"),
        symptom="Aggregate read volume at the source during start-up is several times the artifact size.",
        chain="If the stored format is not shardable, each rank reads the full file and discards everything outside its own shard, so the source serves the artifact once per rank while delivering only one artifact's worth of useful bytes.",
        metric="Bytes served by the source during start-up divided by the artifact size.",
        signature="The ratio equals the parallel degree rather than one, and per-rank read volume equals the full artifact rather than its shard.",
        confounders=(
            "Retries on a flaky path, which also inflate served bytes but irregularly.",
            "Readahead fetching beyond what is used, which inflates volume by a smaller factor.",
            "Several replicas starting at once, which multiplies volume for a different reason.",
        ),
        fixes=(
            "Store the artifact in a shardable layout so each rank reads only its own portion.",
            "Have one rank read and scatter to the others over the interconnect, which is usually faster than the storage path.",
            "Pre-shard the artifact in the build pipeline to match the deployed parallel degree.",
        ),
        rollback="Return to full reads if pre-sharding pins the artifact to one parallel degree that the deployment needs to vary.",
        options=("having one rank read and scatter over the interconnect", "storing the artifact in a shardable layout"),
        tradeoff="whether the interconnect is faster than the storage path for the scatter",
        flip="the parallel degree needs to vary across deployments, at which point a pre-sharded layout stops fitting and read-and-scatter is the flexible option",
        falsifier="source-served bytes equal the artifact size rather than a multiple of it",
        wrong_claim="Each rank only keeps its own shard, so the read volume is one artifact spread across the ranks.",
        wrong_why="What a rank keeps and what it reads are different quantities; with a non-shardable format it reads everything and discards most of it, and the source sees the full read.",
        threshold="Expect source-served bytes to equal the artifact size; investigate any multiple of it.",
        cost="The source is provisioned for a multiple of the volume actually needed, and every rank waits through the extra transfer.",
        scaling="Amplification equals the parallel degree, so it worsens exactly as models grow large enough to require higher degrees.",
        quant=q_read_amplification,
    ),
    Mechanism(
        key="storage_share_bound", topic="storage",
        title="a storage optimisation is bounded by the share of wall time storage occupies",
        concepts=("amdahl", "profiling", "prioritisation"),
        symptom="A large relative improvement in storage throughput produced no measurable change in the service.",
        chain="Storage appears on the critical path only during start-up and checkpoint events, so if those occupy a small share of wall time, even eliminating storage time entirely changes end-to-end results by at most that share.",
        metric="Share of wall time attributable to storage operations, computed before any storage benchmarking begins.",
        signature="The measured storage share bounds the observed end-to-end improvement, and the two agree.",
        confounders=(
            "Page cache making repeat runs skip storage entirely, which collapses the apparent share.",
            "Prefetching overlapping storage with compute, which changes attribution without changing duration.",
            "Restart frequency rising, which raises the share without any change in storage speed.",
        ),
        fixes=(
            "Compute the upper bound on achievable gain before benchmarking the storage path.",
            "Redirect effort to the dominant stage if the bound is below the threshold that would justify the work.",
            "Re-compute the bound if restart or checkpoint frequency changes, since the share moves with them.",
        ),
        rollback="Stop the storage work and record the computed bound as the reason, so the question is not reopened without new information.",
        options=("computing the upper bound before benchmarking", "redirecting effort to the dominant stage"),
        tradeoff="whether the storage share is large enough for an improvement to matter end to end",
        flip="restart or checkpoint frequency rises enough that the storage share becomes material, at which point the bound must be recomputed and the work may be justified",
        falsifier="the storage share of wall time is large enough that the bound exceeds the improvement threshold",
        wrong_claim="We made model loading three times faster, so start-up-sensitive workloads will benefit substantially.",
        wrong_why="The benefit is capped by how much of the workload's wall time loading occupied, and for a long-running replica that share is small enough to make a threefold improvement invisible.",
        threshold="Require the computed storage share to exceed the improvement threshold before storage work is scheduled.",
        cost="Engineering time spent on a stage that cannot move the outcome is the most expensive kind of correct work.",
        scaling="The share falls as replica lifetime grows and rises as restart frequency grows, so the same optimisation is worthwhile in one operating regime and pointless in another.",
        quant=q_storage_share,
    ),
    Mechanism(
        key="artifact_integrity", topic="storage",
        title="an unverified artifact fails silently as degraded quality rather than loudly as an error",
        concepts=("integrity", "checksums", "reliability"),
        symptom="One replica produces measurably worse output than its identically configured peers and reports no errors.",
        chain="A partially corrupted artifact can still parse and load, so the replica starts normally and serves with damaged weights, and nothing in the serving path is positioned to notice.",
        metric="Content hash of the loaded artifact verified against the published hash, per replica.",
        signature="The hash differs on the affected replica while every configuration value matches its peers.",
        confounders=(
            "A different artifact version deployed to that replica, which also produces divergent output.",
            "Non-deterministic reduction order, which produces small divergence without any corruption.",
            "A hardware memory fault, which corrupts after load rather than during transfer.",
        ),
        fixes=(
            "Verify a content hash after transfer and refuse to start on mismatch.",
            "Record the verified hash in the replica's identity so divergence is attributable later.",
            "Compare output against a fixed probe set at start-up as a second, cheaper check.",
        ),
        rollback="Take the replica out of service on hash mismatch rather than restarting it, since a restart re-fetches from the same path that produced the corruption.",
        options=("verifying a content hash after transfer", "comparing output against a fixed probe set at start-up"),
        tradeoff="whether the corruption occurs before load, where a hash catches it, or after, where it does not",
        flip="corruption turns out to occur in device memory after load, at which point a transfer hash passes and only the output probe detects it",
        falsifier="the artifact hash matches on the affected replica",
        wrong_claim="The replica started successfully and is serving traffic, so the model loaded correctly.",
        wrong_why="Successful parsing is not integrity; a corrupted tensor of the right shape loads without complaint and degrades output silently.",
        threshold="Require artifact hash verification before a replica is allowed to accept traffic.",
        cost="A silently degraded replica serves damaged output for as long as it takes someone to notice, which is usually far longer than a start-up failure would have taken.",
        scaling="Corruption probability grows with artifact size and transfer count, so larger models across larger fleets encounter it more often.",
        quant=q_integrity,
    ),
)
