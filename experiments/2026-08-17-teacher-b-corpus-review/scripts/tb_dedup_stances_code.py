"""Authored teacher-B exemplars for the deduplicating stage, code families 504-513.

Each entry is (head, body, quality_dimensions, risks, evidence_required, confidence).
"""

QD = (2, 2, 2)


def _body(mech, hyp, falsi, metrics, est, exp, conf, roll):
    return f"""Mechanism. {mech}

Falsifiable hypothesis. H1: {hyp} Falsified if {falsi}

Metrics. {metrics} {est}

Controlled experiment. {exp}

Confounders. {conf}

Rollback criteria. {roll}"""


FAM_504 = [
 ("STANCE 170 - Compute the KV footprint in exact integer bytes and refuse to fold the two-tensor factor into an unexplained constant.",
  _body(
   "A decode step stores one key and one value vector per attention layer per position, so the byte count is layers times sequence length times key-value heads times head dimension times bytes per value, multiplied by two for the key and value tensors. Writing that factor of two as a literal with a named constant keeps the formula auditable; hiding it inside a magic multiplier is the defect that makes these estimators silently wrong by a factor of two.",
   "the estimator's output equals the allocator's reported per-request cache reservation within the padding the runtime documents, so the formula is complete.",
   "the two differ by a constant ratio, which localises the error to a missing dimension factor rather than to rounding.",
   "Predicted bytes per request from the formula, allocator-reported reservation per request, ratio of the two across several sequence lengths, and the residual after removing block padding.",
   "The allocator reservation is MEASURED from the runtime; the formula output is an ESTIMATE and must be labelled as such wherever it is surfaced.",
   "Evaluate the function against the runtime's own accounting at several layer counts, head configurations and sequence lengths, including a grouped-query configuration where key-value heads differ from attention heads, since that is where an incorrect head term hides.",
   "Grouped-query attention makes key-value heads smaller than attention heads, so using the wrong head count inflates the estimate. Quantised caches change bytes per value. Block-based allocators round up, so exact agreement is not expected and the padding term must be modelled separately.",
   "Reject non-positive inputs with a raised error rather than returning zero, because a zero footprint silently passes downstream capacity checks. If the estimator disagrees with the allocator by more than the documented padding, stop using it for admission decisions until the discrepancy is explained.",
  )),
 ("STANCE 171 - Validate the arguments before computing, because a negative or zero dimension produces a plausible number that fails silently downstream.",
  _body(
   "Every argument is a physical count and must be a positive integer. A zero sequence length yields zero bytes, which reads as a valid answer and will be accepted by any caller that only checks for exceptions. Raising on non-positive input converts a silent capacity miscalculation into an immediate, localised failure at the boundary where the bad value entered.",
   "the function raises for every non-positive argument and for non-integer arguments, so no caller can obtain a misleading zero or fractional footprint.",
   "any non-positive input returns a value, which means the guard is incomplete and the failure will surface later as an over-admission.",
   "Count of rejected inputs by argument, presence of a raised error for each boundary value, and absence of floating-point results in the returned type.",
   "These are MEASURED directly by unit tests; no ESTIMATE is involved, which is why this property is worth asserting rather than reasoning about.",
   "Enumerate boundary cases per argument at zero, at minus one and at a non-integer value, asserting the specific exception type rather than merely that something was raised, since a TypeError from arithmetic is not the same contract as an explicit ValueError.",
   "Booleans are integers in Python and will pass a naive integer check, so True as a head count must be considered. Values arriving from configuration files are strings and may coerce silently if the guard is written loosely.",
   "Keep the guard at the function boundary rather than in the caller, so that every call site inherits it. If a caller needs a permissive path, it must pass validated values, not disable the check.",
  )),
 ("STANCE 172 - Return integer bytes and keep the arithmetic in integers, because floating point silently loses exactness at realistic cache sizes.",
  _body(
   "Cache footprints reach the order of billions of bytes. Once the computation passes through a float, the result is no longer exact and comparisons against allocator limits become approximate near the boundary. Python integers are unbounded, so keeping every operand and the return value integral costs nothing and preserves exactness for admission arithmetic.",
   "the returned value is an integer for all valid inputs and equals the product computed with exact arithmetic, so no precision is lost at admission-boundary magnitudes.",
   "the function returns a float for some input, which shows a division or a float constant has entered the expression.",
   "Returned type across the input sweep, exact-equality comparison against an independently computed integer product, and the largest input at which agreement holds.",
   "Both sides here are MEASURED by test rather than ESTIMATE, and any conversion to gigabytes for display must happen at the presentation layer only.",
   "Assert the return type and exact equality against a reference integer product over a sweep that includes large layer counts and long sequences, and separately assert that unit conversion helpers do not feed their float results back into the estimator.",
   "A division by a byte-per-value constant expressed as a float will convert the whole expression. Formatting helpers that round to gigabytes can be mistaken for the estimator itself if both are exported from the same module.",
   "If a float is required by a downstream interface, convert at that interface and keep the exact value internally. Revert any change that introduces division into the core expression until an exactness test covers it.",
  )),
]

FAM_505 = [
 ("STANCE 173 - State the divisibility contract explicitly, because tensor parallelism partitions a fixed set of ranks and a remainder has no valid placement.",
  _body(
   "A tensor-parallel group of size tp partitions each layer's weights across tp ranks, and the world is divided into world_size divided by tp such groups. If tp does not divide world_size, some ranks belong to an incomplete group and collectives over that group cannot be formed. The validator must therefore reject the remainder case rather than truncating, since truncation produces a launch that fails later inside a collective with an unreadable error.",
   "every world_size and tp pair the validator accepts satisfies exact divisibility and both are positive integers, so no accepted configuration can produce an incomplete group.",
   "any accepted pair leaves a remainder, which means the check was written as a comparison rather than as a modulus test.",
   "Count of accepted and rejected pairs over an exhaustive small grid, remainder for every accepted pair, and agreement with the launcher's own acceptance on the same grid.",
   "Validator outcomes are MEASURED exhaustively over the small grid; agreement with a distributed launcher at large world sizes is an ESTIMATE until it is run at that size.",
   "Enumerate every pair over a bounded grid, assert acceptance exactly when the remainder is zero and both values are positive, and cross-check a sample of accepted and rejected pairs against an actual launch to confirm the validator is not merely self-consistent.",
   "Some frameworks also require tp to divide the attention head count, which is a separate constraint the validator may not model and must not be assumed to cover. Environment variables may override world size after validation.",
   "Reject rather than silently adjusting tp to the nearest divisor, because a silent adjustment changes the parallelism plan without the operator's knowledge. If the launcher accepts a pair the validator rejects, treat the validator as the stricter contract and reconcile before relaxing it.",
  )),
 ("STANCE 174 - Reject non-positive and non-integer values before the modulus, because the remainder test is meaningless on them.",
  _body(
   "A modulus operation on zero raises, and on a negative value it returns a result that can be zero, so a validator that tests divisibility first will either crash or accept a nonsensical configuration such as a negative tp. Ordering the type and positivity guards ahead of the arithmetic makes the failure mode explicit and the error message actionable.",
   "the validator raises or returns false for zero, negative and non-integer inputs before evaluating any remainder, so no arithmetic exception can escape it.",
   "a zero tp produces a division error rather than the declared rejection, which shows the guards are ordered after the arithmetic.",
   "Outcome and exception type for each boundary input, and confirmation that no unhandled arithmetic exception appears for any input in the boundary set.",
   "These outcomes are MEASURED by unit test; the claim that production configuration sources cannot emit such values is an ESTIMATE and is exactly the assumption the guard exists to defend.",
   "Test zero, negative, float and string inputs for both arguments and assert the declared behaviour, distinguishing a returned false from a raised error since callers branch differently on the two.",
   "Booleans satisfy an integer check and would accept tp of one under True. Configuration parsers may supply strings that coerce under comparison but not under modulus.",
   "Choose one contract, either returning false or raising, and keep it uniform across boundary and divisibility failures, since a mixed contract forces every caller to handle both paths and will be handled inconsistently.",
  )),
 ("STANCE 175 - Test the validator against the parallelism plan it authorises rather than against its own restatement of the rule.",
  _body(
   "A validator that is tested only against a reimplementation of its own condition proves nothing beyond internal consistency. The property that matters is whether every accepted configuration launches and every rejected one would have failed. Binding the test to an observable launch outcome is what makes the validator evidence rather than decoration.",
   "for a bounded set of configurations, acceptance by the validator predicts a successful group formation and rejection predicts a failure, so the validator is a faithful proxy for launchability.",
   "some accepted configuration fails to form groups, which means the validator's rule is incomplete rather than merely mis-implemented.",
   "Per-configuration validator outcome, launch outcome, group count formed, and the set of configurations where the two disagree.",
   "Launch outcomes are MEASURED at the sizes actually run; extrapolation of the rule to larger world sizes than were tested is an ESTIMATE and must be recorded as untested.",
   "Run the bounded configuration set through both the validator and a minimal launch that only forms the process groups and exits, and report the disagreement set explicitly rather than a pass rate, because a single disagreement is the finding.",
   "Available device count bounds which configurations can be launched, so the tested set is smaller than the validated set. Launch failures can arise from unrelated environment problems and must be distinguished from group-formation failures.",
   "If any disagreement is found, tighten the validator rather than the test, and record which world sizes the rule has actually been exercised at so that later scale-ups do not inherit an unverified guarantee.",
  )),
]

FAM_506 = [
 ("STANCE 176 - Treat the parser as a trust boundary: model output is untrusted input and must be validated structurally before any field is read.",
  _body(
   "A tool call arrives as model-generated text and is used to dispatch an action. Any field read before validation is an unvalidated value reaching an executor. The parser must therefore complete structural validation, that the payload is an object, that the name is a string and the arguments are an object, before returning anything a caller could act on, rather than validating fields lazily as they are accessed.",
   "no code path returns a partially validated call, so every value a caller receives has passed the full structural check.",
   "any error path returns a call object with some fields populated, which means a caller can act on unvalidated input.",
   "Count of returned objects by path, presence of populated fields on error paths, and rejection counts by violated rule.",
   "These are MEASURED by construction tests over malformed payloads; the claim that upstream generation constrains the format is an ESTIMATE that a grammar-constrained decoder makes plausible but does not guarantee.",
   "Feed a corpus of malformed payloads covering non-object roots, non-string names, non-object arguments and truncated JSON, and assert that each is rejected with a distinguishable reason and that nothing actionable is returned.",
   "A schema-constrained decoder makes malformed output rare in testing, which biases a corpus drawn from production toward well-formed cases. Retries can mask rejection rates.",
   "Reject rather than repair, because a repaired call is a call the model did not make. If rejection rates rise after a model change, treat that as a signal about the model rather than a reason to loosen the parser.",
  )),
 ("STANCE 177 - Reject duplicate keys explicitly, because standard JSON decoders keep the last occurrence and silently discard an earlier conflicting value.",
  _body(
   "Most JSON parsers accept repeated object keys and retain the final one. If a payload contains a tool name twice with different values, the parser silently selects one and the discarded value is invisible to every downstream check. Detecting duplicates requires a hook at decode time, such as an object-pairs handler, because by the time a dictionary exists the evidence has been destroyed.",
   "the parser rejects any payload containing a repeated key at any nesting level, so no value can be silently dropped by last-wins resolution.",
   "a duplicated key is accepted and resolved to the last value, which shows the check runs after decoding rather than during it.",
   "Rejection outcome for payloads with duplicated keys at the root and inside arguments, and confirmation that the detection occurs during decoding rather than on the resulting dictionary.",
   "Detection behaviour is MEASURED against a crafted corpus; the frequency of duplicate keys in real model output is an ESTIMATE and is not the basis for the check, since the consequence rather than the rate justifies it.",
   "Construct payloads with duplicates at the root and in nested argument objects, verify rejection for both, and confirm by inspection that the decode hook is installed rather than relying on a post-decode key count that cannot see the duplication.",
   "A post-decode length comparison cannot detect duplicates because the dictionary already collapsed them. Streaming or incremental parsers may not expose a pairs hook, which changes the available mechanism.",
   "If the decoder cannot expose duplicate keys, state that limitation in the parser's contract rather than claiming a guarantee it does not provide. Do not substitute a post-decode check and describe it as duplicate detection.",
  )),
 ("STANCE 178 - Reject unknown top-level fields rather than ignoring them, because an ignored field is an instruction the caller believes was honoured.",
  _body(
   "Ignoring unrecognised fields is tolerant for data interchange and unsafe for dispatch. If a payload carries an unexpected top-level field, either the model is emitting a format the parser does not model or something is injecting structure. Both are conditions the caller should see. Strict rejection converts a silent divergence into a reportable event at the point it occurs.",
   "any top-level field outside the declared set causes rejection with the offending field named, so no unmodelled instruction passes through unnoticed.",
   "unknown fields are dropped and the call is dispatched, which means format drift will accumulate without any signal.",
   "Rejection count by unknown field name, the declared allowed set, and the rate of unknown-field rejections over time as a drift indicator.",
   "Rejections are MEASURED; the interpretation of a rising rate as model drift rather than as an upstream change is an ESTIMATE requiring the deployment timeline to confirm.",
   "Add payloads carrying one unexpected top-level field each, assert rejection and that the message names the field, and confirm that adding a legitimate new field requires an explicit change to the allowed set rather than passing by default.",
   "Some providers add metadata fields as part of routine format evolution, so strictness creates a maintenance obligation that must be accepted deliberately. Nested argument schemas are tool-specific and are a separate concern from the top-level envelope.",
   "Keep the allowed set as an explicit declaration in one place so that loosening it is a reviewable change. If strictness causes breakage after a provider update, widen the set deliberately rather than switching the parser to a permissive mode.",
  )),
]

FAM_507 = [
 ("STANCE 179 - Bound the total attempt count and the total elapsed time, because a bound on attempts alone does not bound the caller's latency.",
  _body(
   "Exponential backoff makes later waits dominate the total, so a limit expressed only in attempts permits an unbounded-looking delay from the caller's perspective. A retry policy that a caller can reason about needs both a maximum attempt count and a deadline, with the deadline checked before each sleep so that a wait which would overrun it is not taken at all.",
   "no retried operation exceeds the configured deadline, and the attempt count never exceeds its maximum, under any combination of failure timings.",
   "some run overruns the deadline, which indicates the deadline is checked after sleeping rather than before.",
   "Total elapsed time per retried operation against its deadline, attempts used, sleep time against work time, and the count of operations terminated by deadline rather than by attempt exhaustion.",
   "Elapsed times are MEASURED under injected failures; the worst-case total derived from the backoff schedule is an ESTIMATE that the measurement must confirm.",
   "Inject failures at each attempt index and at varying operation latencies, then assert the elapsed bound holds in every case, including the case where the underlying call itself is slow enough to consume the deadline without any retries.",
   "Timer resolution and scheduler delay add to sleeps, so a bound asserted at exactly the nominal total will be flaky. Nested retries at multiple layers multiply attempts and are the usual cause of an unexpected total.",
   "Prefer reducing the deadline over reducing attempts when latency must be recovered, since the deadline is the property callers depend on. If retries are nested, remove the inner layer rather than tuning both.",
  )),
 ("STANCE 180 - Cap the backoff and add jitter, because synchronised clients retry in phase and reconstruct the load that caused the failure.",
  _body(
   "When many clients fail simultaneously, an unjittered schedule makes them all retry at the same offsets, producing a retry wave at exactly the moment the dependency is trying to recover. Capping the interval bounds the tail wait and full jitter spreads the arrivals, converting a synchronised burst into a smoother arrival process.",
   "with jitter enabled, the retry arrival rate at a recovering dependency has a substantially lower peak than the unjittered schedule at the same client count and failure pattern.",
   "peak arrival rate is unchanged, which would indicate the jitter is applied to an already-dominant fixed component or that arrivals are synchronised by something other than the schedule.",
   "Retry arrivals per unit time at the dependency, peak-to-mean arrival ratio, distribution of realised wait intervals, and recovery time after the fault is cleared.",
   "Arrival rates are MEASURED in the harness; the effect on a production dependency's recovery is an ESTIMATE until observed under a real incident or a controlled fault injection.",
   "Run many simulated clients against an instrumented dependency that fails for a fixed window, comparing arrival profiles with jitter disabled, with the interval capped only, and with both, so the two mechanisms are separated rather than credited jointly.",
   "Client start times may already be staggered by deployment, which reduces the apparent benefit. A retry budget or circuit breaker elsewhere in the stack can dominate the arrival shape and mask the schedule's contribution.",
   "Revert to the recorded prior schedule if recovery time worsens, and change the cap and the jitter independently so the responsible parameter is identifiable. Do not remove jitter to make latency more predictable, since predictable retry timing is the failure mode.",
  )),
 ("STANCE 181 - Classify errors before retrying, because retrying a deterministic failure converts one error into several at no benefit.",
  _body(
   "Retries are only useful for transient conditions. A malformed request, an authorisation failure or a schema violation will fail identically on every attempt, so retrying multiplies load and delays the caller's error without any chance of success. The policy therefore needs an explicit predicate identifying retryable conditions, defaulting to not retrying when the classification is unknown.",
   "no error classified as non-retryable is retried, and the default for an unrecognised error is a single attempt, so unknown failures cannot amplify.",
   "unrecognised errors are retried by default, which means the predicate is expressed as a non-retryable deny list rather than a retryable allow list.",
   "Attempts per error class, count of retries on non-retryable classes, share of total attempts attributable to unclassified errors, and caller-visible latency by class.",
   "Attempt counts are MEASURED from injected error classes; the completeness of the classification against production error surfaces is an ESTIMATE that must be revisited whenever a dependency changes.",
   "Inject each known error class and assert the attempt count for each, including at least one deliberately unrecognised error to confirm the default is a single attempt rather than the full schedule.",
   "Transport errors can wrap application errors so that class is only visible after unwrapping. Timeouts are ambiguous because the operation may have succeeded, which makes them a retry-safety question rather than a classification question.",
   "Default to not retrying when classification is uncertain, and add classes explicitly as they are understood. If a non-retryable class is found being retried, fix the predicate rather than lowering the attempt count, because the count is not the defect.",
  )),
]

FAM_508 = [
 ("STANCE 182 - Declare the percentile definition, because nearest-rank and interpolated methods disagree on small samples and the disagreement is mistaken for a regression.",
  _body(
   "There are several accepted definitions of a percentile and they differ on finite samples, most visibly at high percentiles on small windows. If the definition is not stated, a change of library or window size shifts the reported value without any change in the underlying latency, and that shift will be investigated as a performance regression.",
   "the function's output matches a stated reference definition exactly on fixed inputs, so any change in reported value reflects a change in data rather than in method.",
   "the output matches neither the nearest-rank nor the stated interpolated reference, which means the implementation is a third undocumented method.",
   "Computed percentile against the reference definition on fixed vectors, the maximum divergence between definitions at the window sizes in use, and the sample count backing each reported value.",
   "Agreement with the reference is MEASURED on fixed inputs; the divergence expected at production window sizes is an ESTIMATE derived from those windows and should be published alongside the metric.",
   "Compare the implementation against both candidate definitions on hand-checked vectors of length one, two and several, and record which definition the metric contract adopts in the same place the metric is documented.",
   "Aggregation across shards averages percentiles, which is not a percentile of the combined distribution and dominates any definitional difference. Sampling upstream changes the effective sample size invisibly.",
   "Do not change the definition to make a number look better; if it must change, republish the historical series under the new definition or mark the discontinuity, since a silent redefinition breaks every prior comparison.",
  )),
 ("STANCE 183 - Define the empty and single-element cases explicitly rather than letting them fall out of the arithmetic.",
  _body(
   "An empty latency window is a normal condition during low traffic or after a restart, and an implementation that indexes into an empty sequence will raise inside a metrics path where the exception is often swallowed. Returning a sentinel silently is equally bad because a zero latency reads as an excellent result. The behaviour must be a stated part of the contract.",
   "the empty case produces a distinguishable outcome that no consumer can mistake for a latency value, and the single-element case returns that element for every percentile.",
   "the empty case returns zero or a numeric default, which is indistinguishable from a real measurement of near-zero latency.",
   "Outcome for empty and single-element inputs, count of empty windows in production per interval, and consumer behaviour when the sentinel is received.",
   "The function's outcomes are MEASURED by test; the frequency of empty windows in production is an ESTIMATE from traffic patterns and should be checked against the deployment's actual low-traffic periods.",
   "Assert the empty and single-element behaviour directly, then confirm at least one downstream consumer, such as an alert rule, handles the sentinel as intended rather than treating it as a value.",
   "Alerting systems often coerce missing data to zero, reintroducing the failure downstream of a correct function. Dashboards may interpolate across gaps and hide empty windows entirely.",
   "Prefer an explicit absent value over a numeric sentinel where the consumer supports it. If a numeric default must be used, choose one that cannot be a valid latency and document it where the metric is consumed, not only where it is produced.",
  )),
 ("STANCE 184 - Do not average percentiles across shards or windows; aggregate the underlying observations or use a mergeable sketch.",
  _body(
   "The percentile of a union is not a function of the percentiles of the parts. Averaging per-shard p99 values understates the combined p99 whenever load is uneven, which is precisely when the tail matters. Correct aggregation requires either the raw observations or a summary structure with a defined merge, such as a histogram with fixed bucket boundaries.",
   "the aggregated tail computed by merging distributions differs materially from the average of per-shard tails under uneven load, so the averaging method is understating the real tail.",
   "the two agree under deliberately skewed load, which would indicate the shards are homogeneous enough that the distinction is not currently material.",
   "Merged-distribution p99 against averaged per-shard p99, per-shard load skew, bucket resolution near the tail, and the resulting error attributable to bucketing.",
   "Both aggregations are MEASURED from the same observation set; bucketing error is an ESTIMATE bounded by the bucket widths and must be reported with the metric's resolution.",
   "Replay one observation set through both aggregation paths under uniform and skewed shard loads, and report the gap as a function of skew rather than as a single number, since the gap is what justifies changing the pipeline.",
   "Coarse histogram buckets at the tail impose their own error that can be confused with the aggregation error. Clock skew between shards misaligns windows and adds a separate discrepancy.",
   "Change the aggregation before tuning any threshold that depends on it, and restate historical alert thresholds under the corrected method. If mergeable structures are unavailable, report per-shard tails separately rather than publishing an average and calling it a percentile.",
  )),
]

FAM_509 = [
 ("STANCE 185 - Use exact integer ceiling arithmetic for block counts, because float division rounds the wrong way at exact multiples.",
  _body(
   "The number of blocks needed is the sequence length divided by the block size, rounded up. Computing this by float division and then applying a ceiling can return one block too many when the division is exact but representable only approximately, and can under-count for large values where the float loses precision. The integer form, using floor division of the length plus block size minus one, is exact for all inputs.",
   "the integer expression agrees with an exact reference for every input in a wide sweep including exact multiples and large values, while the float expression diverges on at least one.",
   "the two agree everywhere in the sweep, which would mean the tested range is too small to expose the precision boundary rather than that the float form is safe.",
   "Agreement between the integer expression, the float expression and an exact reference across the sweep, and the smallest input at which the float form diverges.",
   "Agreement is MEASURED exhaustively over the sweep; the claim that production lengths stay below the divergence point is an ESTIMATE that grows unsafe as context limits increase.",
   "Sweep sequence lengths across exact multiples, one below and one above each multiple, and several very large values, comparing all three computations and reporting the first divergence rather than a pass rate.",
   "Small test ranges never reach the precision boundary, which is why this defect survives review. A block size of one masks the rounding question entirely.",
   "Keep the arithmetic integral and reject any change reintroducing division into the expression. Over-allocation by one block per request is not benign at scale, since it reduces admitted concurrency proportionally.",
  )),
 ("STANCE 186 - Validate both arguments, because a zero block size raises and a zero length silently returns a valid-looking zero.",
  _body(
   "The two arguments fail differently. A zero or negative block size makes the division undefined and raises inside the expression, producing an error far from its cause. A zero or negative length returns zero blocks, which passes every downstream check and results in a request admitted with no allocation. Both must be rejected at the boundary with an explicit error.",
   "every non-positive value of either argument is rejected before the arithmetic, so neither a raised division error nor a zero block count can escape the function.",
   "a zero length returns zero rather than raising, which shows the guard covers only the divisor.",
   "Outcome per boundary input for both arguments, and confirmation that no arithmetic exception originates inside the expression for any tested input.",
   "These outcomes are MEASURED by test; whether callers can supply a zero length depends on upstream validation and is an ESTIMATE the guard exists precisely because it cannot be relied upon.",
   "Test zero, negative and non-integer values for both arguments independently, asserting the same explicit error type for each, and confirm the zero-length case raises rather than returning zero.",
   "Truncated or empty prompts can legitimately produce a zero token count upstream, so the caller may need to handle that case before calling rather than relying on the exception as control flow.",
   "Raise rather than clamping to one block, because clamping invents an allocation the caller did not request. If a caller legitimately needs zero, it must branch before the call.",
  )),
 ("STANCE 187 - Report the internal fragmentation the block size creates, because the block count alone hides the memory it wastes.",
  _body(
   "Rounding up to whole blocks means the last block is partly unused. Averaged over many requests that waste is roughly half a block each, which at large block sizes is a significant share of cache capacity and directly reduces admitted concurrency. A planner that reports only block counts gives no visibility into this, so the block size is tuned without seeing its cost.",
   "measured wasted positions per request are close to half the block size under a realistic length distribution, so fragmentation scales with block size as predicted.",
   "waste is far below half a block, which would indicate lengths are aligned to the block size by upstream padding and the model does not apply.",
   "Wasted positions per request, total wasted capacity as a share of cache, admitted concurrency at several block sizes, and the length distribution the waste is computed over.",
   "Waste is MEASURED from the realised length distribution; the half-block rule is an ESTIMATE valid only for lengths spread relative to the block size and must not be quoted where lengths are aligned.",
   "Compute realised waste across the production length distribution at several block sizes and pair it with measured admitted concurrency, so the fragmentation cost and the allocator-efficiency benefit of larger blocks are visible together.",
   "Larger blocks reduce allocator overhead and improve locality, so the comparison is a trade-off rather than a minimisation. Prefix sharing changes effective occupancy independently of block size.",
   "Do not reduce block size on fragmentation evidence alone; require the concurrency measurement at the new size, and revert if allocator overhead or throughput regresses beyond the recovered capacity.",
  )),
]

FAM_510 = [
 ("STANCE 188 - Report every missing variable in one pass, because a checker that fails on the first one turns a single fix into a sequence of restarts.",
  _body(
   "Distributed launches are expensive to start and slow to fail. A checker that raises on the first missing variable forces the operator to discover the requirements one restart at a time. Collecting all violations and reporting them together converts that sequence into a single corrective action, which is the entire operational value of the checker.",
   "for an environment missing several required variables, the checker names all of them in one invocation, so no fix-and-retry cycle is needed to discover the rest.",
   "only the first missing variable is reported, which means the check raises inside the loop rather than accumulating.",
   "Number of missing variables reported per invocation against the number actually missing, and the number of restart cycles required to reach a valid environment.",
   "Report completeness is MEASURED by test; the reduction in operator restart cycles is an ESTIMATE unless launch attempts are counted before and after.",
   "Construct environments missing one, several and all required variables, and assert the reported set equals the missing set exactly rather than merely being non-empty.",
   "Some variables are set by the launcher after the check runs, so a variable missing at check time is not necessarily missing at use time and the check point must be stated. Defaults applied by libraries can make a variable appear present.",
   "Keep the check advisory and separate from the launch path if it risks blocking a valid configuration, and record which variables are required versus merely recommended so the report does not conflate the two.",
  )),
 ("STANCE 189 - Never print values, only names, because distributed environments carry credentials alongside topology settings.",
  _body(
   "A diagnostic that dumps the environment is the most common way tokens reach log aggregation, where they are retained and broadly readable. Reporting only the names of missing or malformed variables preserves all the diagnostic value, since the operator knows the values, while removing the disclosure. The distinction must be enforced in the checker rather than left to a redaction filter downstream.",
   "no code path in the checker emits a variable's value, so its output cannot disclose a secret regardless of what the environment contains.",
   "some path formats a value into a message, typically inside a validation error for a malformed variable, which is where this leak usually appears.",
   "Count of value-emitting code paths found by inspection and by a test that sets a distinctive value and asserts its absence from all output streams.",
   "Absence from output is MEASURED by that test; the completeness of the audit across all error paths is an ESTIMATE and should be supported by a lint rule rather than by review alone.",
   "Set every required and optional variable to a unique sentinel, run the checker across valid, missing and malformed cases, and assert no sentinel appears in stdout, stderr or the raised exception text.",
   "Exception messages from underlying parsers may embed the offending value even when the checker does not. Structured logging can capture context fields that never appear in the formatted message.",
   "If a value must be shown to diagnose a malformed setting, show a fixed-length prefix or a length only, and make that an explicit, reviewed exception rather than a general policy of echoing values.",
  )),
 ("STANCE 190 - Validate the interpretable structure of the values present, not merely that the names are set.",
  _body(
   "A rank set to a non-integer, a world size inconsistent with the rank range, or a master port outside the valid range are all present-but-wrong conditions that a presence check passes and that fail later inside a collective with an error naming neither the variable nor the cause. Checking type and range at the same point as presence keeps the diagnosis local.",
   "each malformed value is rejected with the variable named and the constraint stated, so the failure is diagnosable without reading a collective's internal error.",
   "malformed values pass and surface later inside the runtime, which means only presence is being checked.",
   "Rejection outcome per malformed case, the constraint reported with each, and the share of launch failures that the checker would have caught in retrospect.",
   "Checker outcomes are MEASURED against crafted environments; the retrospective share of real failures caught is an ESTIMATE from an incident sample and depends on that sample being representative.",
   "Craft environments with a non-integer rank, a rank outside the world size, a port outside the valid range and an unresolvable address, and assert each is rejected with the variable named and no value echoed.",
   "Address resolution depends on network state and can fail transiently, so it is a different class of check from a syntactic one and should be reported separately. Some runtimes tolerate values this check would reject.",
   "Keep syntactic checks blocking and environment-dependent checks advisory, since a resolution failure at check time may not persist. If the checker rejects a configuration the runtime accepts, relax the constraint deliberately and record why.",
  )),
]

FAM_511 = [
 ("STANCE 191 - State the threshold and its derivation, because a hard-coded token count is a hidden policy that no one can audit.",
  _body(
   "Classifying a request as prefill-dominated or decode-dominated depends on where the phase cost crosses over, which is a property of the model, the hardware and the batching policy rather than a universal constant. A literal threshold in the classifier encodes one deployment's crossover permanently and will misclassify on any other. The threshold must be a named parameter with a recorded derivation.",
   "the crossover measured on the target deployment differs materially from the hard-coded threshold, so the constant is deployment-specific rather than general.",
   "the measured crossover matches the constant across several model and hardware pairs, which would justify it as a default while still requiring it to be configurable.",
   "Measured prefill and decode time as functions of prompt and generated token counts, the crossover point, the configured threshold, and the misclassification rate against phase timings.",
   "Phase timings and the crossover are MEASURED on the target deployment; a threshold carried over from another deployment is an ESTIMATE and must be labelled as unvalidated until re-measured.",
   "Sweep prompt and generation lengths on the target hardware, record per-phase times, locate the crossover, and compare classifier output against the timing-derived label rather than against intuition.",
   "Continuous batching mixes phases within a step, so a per-request label is an approximation of a step-level property. Prefix caching removes prefill work without changing the prompt length, which breaks the token-count proxy.",
   "Re-measure the threshold on any model or hardware change rather than carrying it forward, and treat classifier output as advisory for scheduling until the misclassification rate is known.",
  )),
 ("STANCE 192 - Validate that the two counts are non-negative integers and define the zero-generation case, because a request that has not yet produced a token is not decode-dominated.",
  _body(
   "A request at admission has zero generated tokens, and a ratio-based classifier will either divide by zero or classify it by default into whichever branch the code falls through to. Since admission is exactly when the classification is used for scheduling, that default determines behaviour for every new request and must be chosen deliberately rather than inherited from the expression's structure.",
   "the classifier assigns zero-generation requests to the prefill class explicitly and rejects negative or non-integer counts, so no request is classified by fall-through.",
   "zero-generation requests reach a ratio computation, which means the case is handled implicitly.",
   "Classification outcome for zero-generation, zero-prompt and negative inputs, and the share of admitted requests that take the zero-generation path.",
   "Outcomes are MEASURED by test; the share of traffic on that path is MEASURED in production and is usually large enough that the default dominates the classifier's aggregate behaviour.",
   "Assert the classification for zero generated tokens, zero prompt tokens and both zero, and confirm the scheduler's behaviour under each, since the classifier's contract only matters through its consumer.",
   "A zero prompt is unusual but reachable through templating errors and should not silently classify. Counts may arrive as strings from a request log and coerce inconsistently.",
   "Choose the zero-generation default explicitly and document it next to the scheduler policy that consumes it. If the default is changed, treat it as a scheduling policy change requiring the same validation as a threshold change.",
  )),
 ("STANCE 193 - Validate the classifier against measured phase time rather than against the token counts it was built from.",
  _body(
   "A classifier tested against a restatement of its own rule is trivially correct and carries no information. The claim worth testing is that its label predicts which phase dominates the request's realised cost. That requires per-request phase timing as ground truth, which is more work to obtain and is why this validation is usually skipped.",
   "the classifier's label agrees with the timing-derived dominant phase on a held-out request set at a rate materially above what the class prior alone would produce.",
   "agreement is no better than the prior, which means the classifier is reproducing the class imbalance rather than discriminating.",
   "Agreement rate against timing-derived labels, the class prior on the same set, the confusion matrix, and the token-count region where disagreements concentrate.",
   "Phase timings are MEASURED per request; any agreement rate reported without a stated prior is an ESTIMATE of unknown value, since a skewed prior makes a high rate meaningless.",
   "Instrument per-request prefill and decode time on a held-out set, derive the dominant phase from the timings, and report the confusion matrix and the prior together rather than a single accuracy number.",
   "Under continuous batching, per-request phase time is attributed rather than directly observed, and the attribution method affects the ground truth. Prefix cache hits shift timings without shifting counts.",
   "If agreement is near the prior, replace the token-count proxy rather than tuning its threshold, since a threshold change cannot rescue a feature that does not discriminate. Record the prior alongside every reported agreement rate.",
  )),
]

FAM_512 = [
 ("STANCE 194 - Preserve input order when reporting the first duplicate, because a set-based implementation returns an arbitrary element.",
  _body(
   "Iterating a set to find duplicates loses the ordering that makes the answer meaningful. The specified result is the first identifier that repeats in input order, which requires a single ordered pass with a seen-set, returning at the moment a repeat is observed. A set-difference approach can identify that duplication exists but not which occurrence came first.",
   "the function returns the earliest repeating identifier in input order for every ordering of a fixed multiset, so the result is determined by the input rather than by iteration order.",
   "different orderings of the same multiset yield inconsistent results relative to their own order, which indicates a set-based implementation.",
   "Returned identifier against the hand-computed first duplicate across permuted inputs, and stability of the result across repeated runs of the same input.",
   "These are MEASURED by test; hash iteration order can appear stable within one process, so a single run is an ESTIMATE of determinism rather than a demonstration of it.",
   "Test several permutations of the same multiset, asserting the order-correct answer for each, and include a case where two identifiers both repeat so that the earliest-repeat rule is actually exercised.",
   "Small integer or interned string keys can iterate in insertion order by coincidence, hiding the defect. Hash randomisation is disabled in some environments, making the flaw non-reproducible locally.",
   "Keep the single ordered pass rather than optimising to a set operation for brevity, and if the specification is genuinely order-independent, change the specification explicitly rather than letting the implementation redefine it.",
  )),
 ("STANCE 195 - Define what a duplicate identifier means before comparing, because normalisation choices change the answer.",
  _body(
   "Whether two identifiers are the same depends on case sensitivity, surrounding whitespace, and whether a prefix or namespace is part of the identity. A detector that compares raw strings implements one policy silently. Stating the normalisation and applying it in one place makes the policy visible and prevents two components from disagreeing about identity.",
   "the detector's normalisation matches the identity rule used by the component that assigns identifiers, so no pair considered identical there is considered distinct here.",
   "the two normalisations differ, which means duplicates can pass undetected between components regardless of the detector's correctness.",
   "Duplicate counts under each candidate normalisation on the same input, the set of pairs whose classification changes, and the assigning component's stated identity rule.",
   "Counts are MEASURED per normalisation; agreement with the assigning component is an ESTIMATE unless that component's rule is documented rather than inferred from its code.",
   "Run the detector under raw, case-folded and trimmed comparison over a production sample, and report the pairs that change classification, since those pairs are the decision rather than the aggregate counts.",
   "Identifiers generated by different clients may follow different conventions, so a single normalisation may be correct for one source and wrong for another. Unicode normalisation forms introduce equality questions that case folding does not cover.",
   "Apply the normalisation in one shared function used by both the assigner and the detector. If they must differ, document the asymmetry, because an undocumented one will be discovered as a duplicate that was never detected.",
  )),
 ("STANCE 196 - Bound the memory the detector holds, because a seen-set over an unbounded stream grows without limit.",
  _body(
   "Scanning a finite list is safe, but the same function applied to a long-running request stream retains every identifier forever. The resulting growth is slow enough to pass testing and to surface later as an out-of-memory condition in a component that appears to do nothing expensive. Either the bounded-input assumption is stated in the contract, or the retention must be explicitly limited by count or by time window.",
   "memory held by the detector is bounded by the configured window rather than by the total number of identifiers processed, under a long synthetic stream.",
   "retained memory grows with total identifiers processed, which means no bound is in effect regardless of what the documentation claims.",
   "Retained entries and process memory against identifiers processed, the configured window, and the duplicate detection rate as a function of window size.",
   "Memory growth is MEASURED under the synthetic stream; the detection loss from a finite window is an ESTIMATE depending on the duplicate arrival gap distribution and must be measured on real traffic before the window is fixed.",
   "Run a long synthetic stream with a known duplicate gap distribution at several window sizes, recording both retained memory and missed duplicates, so the trade-off is visible rather than assumed.",
   "Duplicates that arrive far apart are missed by any bounded window, so a window chosen on memory grounds alone silently reduces detection. Garbage collection timing makes short memory measurements noisy.",
   "State the bounded-input assumption in the function's contract if no window is implemented, so that a caller applying it to a stream is making a visible mistake. If a window is added, size it from the measured gap distribution rather than from a memory target alone.",
  )),
]

FAM_513 = [
 ("STANCE 197 - Return the assumptions with the number, because a bare capacity estimate is used as a commitment.",
  _body(
   "A planner that returns a single figure will have that figure quoted in a capacity decision without its context. The assumptions, on sequence length distribution, on concurrency, on cache overhead and on the model configuration, determine the answer more than the arithmetic does. Returning them alongside the estimate makes the number falsifiable and makes an out-of-date assumption visible when the workload changes.",
   "every assumption whose change alters the estimate by a material amount is present in the returned assumption set, so no consequential input is hidden.",
   "a sensitivity sweep identifies an input with large influence that is absent from the returned set, which means the disclosure is incomplete.",
   "One-at-a-time sensitivity of the estimate to each input, the returned assumption set, and the gap between influential inputs and disclosed ones.",
   "Sensitivities are MEASURED by sweeping the planner; the estimate itself is an ESTIMATE and must be labelled so wherever it is reported, including in any document that quotes it.",
   "Sweep each input across its plausible range, rank inputs by their influence on the output, and confirm the returned set covers the influential ones rather than the ones that happened to be parameters.",
   "Inputs interact, so one-at-a-time sensitivity understates the influence of correlated parameters such as concurrency and sequence length. The plausible range itself is an assumption.",
   "Do not publish the estimate without the assumption set attached in the same artefact. If the workload moves outside a stated assumption range, treat the estimate as expired rather than adjusting it informally.",
  )),
 ("STANCE 198 - Be conservative in a stated direction, because an estimate that is unbiased in expectation still breaches capacity half the time.",
  _body(
   "Capacity planning is asymmetric: under-provisioning causes rejected requests or out-of-memory failures, while over-provisioning costs money. A central estimate is therefore the wrong target. The planner should state which direction it errs in and by how much, so that the margin is a decision rather than an accident of the arithmetic.",
   "the planner's output exceeds realised usage in a stated high fraction of a validation sample, so its conservatism is quantified rather than asserted.",
   "realised usage exceeds the estimate more often than the stated fraction, which means the margin is smaller than claimed and the label is misleading.",
   "Estimated against realised peak usage per validation case, the exceedance fraction, the margin distribution, and the cost of the unused headroom.",
   "Realised usage is MEASURED; the exceedance fraction on future workloads is an ESTIMATE valid only while the workload distribution matches the validation sample.",
   "Compare the estimate against realised peak usage over a validation sample spanning the workload's variation, and report the exceedance fraction and the margin distribution rather than a mean error.",
   "A validation sample drawn from a period without incidents excludes the tail the margin exists to cover. Autoscaling changes realised usage and can make the estimate appear accurate by adapting to it.",
   "If the exceedance fraction falls below its stated target, increase the margin rather than reinterpreting the label. Retire the estimate when the workload distribution shifts, since a margin validated on one distribution carries no guarantee on another.",
  )),
 ("STANCE 199 - Model the components that do not scale with the estimate's main term, because fixed overheads dominate at small deployments and are omitted first.",
  _body(
   "Capacity planners usually model weights and cache and stop there. Framework overhead, activation workspace, fragmentation, communication buffers and the runtime's own allocations are roughly fixed per process and are a large share of a small device's memory. Omitting them produces an estimate that is accurate at large scale and badly optimistic exactly where headroom is tightest.",
   "the residual between estimated and realised memory is approximately constant across model sizes, identifying an unmodelled fixed overhead rather than a proportional error.",
   "the residual scales with the main term, which points to an error in the weight or cache arithmetic instead.",
   "Estimated against realised memory across several model sizes and device counts, the residual and its relationship to the main term, and the measured baseline footprint of an idle process.",
   "Realised and baseline footprints are MEASURED; the decomposition of the residual into named overheads is an ESTIMATE until each is instrumented individually.",
   "Measure the idle process footprint before any request, then the footprint under load, across several model sizes, and regress the residual against the main term to distinguish a fixed offset from a proportional error.",
   "Allocator caching retains freed memory, so the reported footprint exceeds live usage and inflates the apparent overhead. Graph capture and warm-up allocate persistently and must be included in the baseline.",
   "Add the fixed term explicitly rather than inflating the margin to absorb it, because an absorbed overhead scales incorrectly. Re-measure the baseline on every runtime upgrade, since it is a property of the framework rather than of the model.",
  )),
]

RISKS = ["Generated review content is provisional and is not expert-verified gold.",
         "Recommendations are contingent on the deployment being measured rather than assumed."]
EVIDENCE = ["Measurements collected on the target deployment.",
            "Explicit labelling of estimated versus measured quantities."]
CONFIDENCE = 0.72

STANCES = {
    fam: [(head, body, QD, list(RISKS), list(EVIDENCE), CONFIDENCE) for head, body in entries]
    for fam, entries in (
        (504, FAM_504), (505, FAM_505), (506, FAM_506), (507, FAM_507), (508, FAM_508),
        (509, FAM_509), (510, FAM_510), (511, FAM_511), (512, FAM_512), (513, FAM_513),
    )
}
