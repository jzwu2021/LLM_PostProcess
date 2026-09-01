"""Fine-tuning and post-training mechanisms (topic: training)."""
from __future__ import annotations

from core import Mechanism, Quant, Setting, fmt_int, gib, register


def q_token_budget(s: Setting) -> Quant:
    rows = 3000
    seq = 768
    grad_accum = 4
    micro = 1
    per_step = seq * grad_accum * micro * max(s.gpu_count, 1)
    steps = max(rows * seq // max(per_step, 1), 1)
    return Quant(
        label="how many optimiser steps a corpus of this size actually produces",
        steps=[
            f"Corpus of {fmt_int(rows)} records at {seq} tokens each = {fmt_int(rows * seq)} tokens",
            f"Tokens per optimiser step = seq {seq} * grad_accum {grad_accum} * micro {micro} "
            f"* {s.gpu_count} ranks = {fmt_int(per_step)}",
            f"Steps per epoch = {fmt_int(rows * seq)} / {fmt_int(per_step)} = {steps}",
            f"A run stopped at 75 steps therefore covers "
            f"{min(75 / max(steps, 1), 1.0) * 100:.0f}% of one epoch",
        ],
        value=f"{steps} steps per epoch; a 75-step run sees "
              f"{min(75 / max(steps, 1), 1.0) * 100:.0f}% of the data",
        interpretation=(
            "Step count and data coverage are different quantities. A run that stops early has not "
            "trained on most of the corpus, and any conclusion about the corpus is unsupported."),
    )


def q_lr_scale(s: Setting) -> Quant:
    base_bs = 8
    grad_accum = 4
    eff = base_bs * grad_accum * max(s.gpu_count, 1)
    return Quant(
        label="the effective batch size this configuration produces",
        steps=[
            f"Per-device micro-batch {base_bs}",
            f"Gradient accumulation {grad_accum}",
            f"Ranks {s.gpu_count}",
            f"Effective batch = {base_bs} * {grad_accum} * {s.gpu_count} = {fmt_int(eff)} sequences",
        ],
        value=f"an effective batch of {fmt_int(eff)} sequences",
        interpretation=(
            "Learning rate must be reconciled against this number rather than against the per-device "
            "batch. Changing rank count silently changes the effective batch and therefore the "
            "appropriate rate."),
    )


def q_masking(s: Setting) -> Quant:
    prompt = 500
    answer = 268
    total = prompt + answer
    return Quant(
        label="the share of a training sequence that should contribute loss",
        steps=[
            f"A typical record is about {prompt} prompt tokens and {answer} answer tokens",
            f"Total sequence {total} tokens",
            f"If the prompt is not masked, {prompt / total * 100:.0f}% of the loss comes from "
            f"predicting the prompt",
            f"The model is then optimised largely for reproducing inputs rather than producing answers",
        ],
        value=f"{answer / total * 100:.0f}% of tokens should carry loss, {prompt / total * 100:.0f}% should be masked",
        interpretation=(
            "Prompt masking is not a refinement. Without it the majority of the gradient signal comes "
            "from a task nobody wants the model to perform."),
    )


def q_memory_training(s: Setting) -> Quant:
    w = s.weight_bytes
    grads = w
    optim = w * 2
    return Quant(
        label="the training memory requirement beyond the weights themselves",
        steps=[
            f"Weights: {gib(w)}",
            f"Gradients, same shape as weights: {gib(grads)}",
            f"Two-moment optimiser state: {gib(optim)}",
            f"Subtotal before activations: {gib(w + grads + optim)} across {s.gpu_count} devices",
        ],
        value=f"{gib(w + grads + optim)} of persistent state before any activation memory",
        interpretation=(
            "Training memory is roughly four times the inference footprint before activations are "
            "counted. Sizing a training job from the served artifact underestimates it by a factor "
            "that grows with the optimiser's state."),
    )


def q_overfit_signal(s: Setting) -> Quant:
    return Quant(
        label="why a falling training loss says nothing about capability",
        steps=[
            "Training loss measures fit to the tokens being optimised",
            "A corpus with repeated content lets loss fall by memorising the repetition",
            "Held-out loss on the same distribution falls too, since it shares the repetition",
            "Only a differently distributed evaluation separates memorisation from capability",
        ],
        value="training and held-out loss can both fall while capability is unchanged",
        interpretation=(
            "Held-out loss is only informative if the held-out set is genuinely different. Splitting "
            "a repetitive corpus produces two halves that share the repetition and validate nothing."),
    )


def q_catastrophic(s: Setting) -> Quant:
    return Quant(
        label="the capability surface a narrow fine-tune can disturb",
        steps=[
            "Fine-tuning updates weights shared by every capability the model has",
            "The corpus covers one domain; the weights serve all of them",
            "Gradient steps that improve the target domain can degrade untargeted ones",
            "Nothing in the training signal measures the untargeted capabilities",
        ],
        value="all capabilities share the weights; only one is measured during training",
        interpretation=(
            "Regression on general capability is invisible unless it is deliberately measured. A "
            "general-capability probe belongs in the acceptance criteria, not in a follow-up."),
    )


def q_seq_len_truncation(s: Setting) -> Quant:
    seq = 768
    typical = 900
    return Quant(
        label="what a training sequence length below the data length discards",
        steps=[
            f"Configured training sequence length is {seq} tokens",
            f"A record of {typical} tokens is truncated to {seq}",
            f"The discarded {typical - seq} tokens are at the end, where the answer usually concludes",
            "The model is then trained to produce answers that stop mid-argument",
        ],
        value=f"{typical - seq} tokens discarded per over-length record, taken from the answer's end",
        interpretation=(
            "Truncation during training teaches the truncated behaviour. It is more damaging than "
            "truncation during evaluation, because it changes the model rather than the measurement."),
    )


def q_data_ordering(s: Setting) -> Quant:
    rows = 3000
    repair = 80
    return Quant(
        label="whether a small added subset is reached within the run",
        steps=[
            f"A corpus of {fmt_int(rows)} records with {repair} appended at the end",
            f"Appending places them in the final {repair / rows * 100:.1f}% of the epoch",
            "A run stopped before the epoch completes never reaches them",
            f"Interleaving one per {rows // repair} records spreads them across the whole run",
        ],
        value=f"{repair} appended records sit in the last {repair / rows * 100:.1f}% of the epoch",
        interpretation=(
            "Data ordering decides what a truncated run actually trains on. Appending a targeted "
            "subset and stopping early trains on everything except the subset that motivated the run."),
    )


def q_contamination(s: Setting) -> Quant:
    return Quant(
        label="why repair data authored from benchmark inspection cannot be evaluated on that benchmark",
        steps=[
            "The benchmark's failures were inspected to decide what the repair data should teach",
            "The repair data therefore encodes information about the benchmark's specific items",
            "Training on it and evaluating on that benchmark measures the transfer of that information",
            "The result cannot distinguish learned capability from encoded answers",
        ],
        value="the benchmark is contaminated the moment it is used to author training data",
        interpretation=(
            "Contamination here is a process property, not a text-overlap property. No string-matching "
            "check will detect it, and the only remedy is a benchmark the authoring process never saw."),
    )


def q_checkpoint_selection(s: Setting) -> Quant:
    return Quant(
        label="how many comparisons a checkpoint sweep silently performs",
        steps=[
            "Suppose checkpoints are saved every 25 steps across a 75-step run",
            "That is three candidates, each compared against the baseline",
            "Selecting the best of three against one evaluation inflates the apparent gain",
            "The inflation grows with the number of candidates considered",
        ],
        value="three candidates compared on one evaluation set, best-of-three reported as the result",
        interpretation=(
            "Selecting a checkpoint on an evaluation set makes that set part of training. The reported "
            "gain includes the selection, and only a held-out set can separate them."),
    )


register(
    Mechanism(
        key="step_count_vs_coverage", topic="training",
        title="a step count is not a data coverage figure, and a short run sees little of the corpus",
        concepts=("training", "epochs", "data_coverage"),
        symptom="A fine-tune run completed its configured steps and the effect of a specific part of the corpus is absent.",
        chain="Tokens consumed per optimiser step are the product of sequence length, micro-batch, accumulation and rank count, so a run stopped at a fixed step count consumes a fraction of the corpus determined by that product rather than by intent.",
        metric="Tokens consumed by the run divided by corpus tokens, computed before the run rather than after.",
        signature="Consumed tokens fall well short of one epoch, and the unreached portion corresponds exactly to the part of the corpus whose effect is missing.",
        confounders=(
            "Shuffling changing which records are reached, which alters coverage without changing the fraction.",
            "Packing several records per sequence, which changes tokens per record.",
            "Rank count differing from the configuration used to compute the estimate.",
        ),
        fixes=(
            "Compute steps per epoch from the configuration and state the intended coverage before launching.",
            "Shuffle so a truncated run samples the corpus uniformly rather than its prefix.",
            "Raise the step count to reach the intended coverage rather than inferring the effect from a partial run.",
        ),
        rollback="Discard conclusions drawn from a run whose coverage was below the intended fraction, rather than extrapolating them.",
        options=("computing and stating intended coverage before launching", "shuffling so a truncated run samples uniformly"),
        tradeoff="whether the run is long enough to reach the data the conclusion depends on",
        flip="the corpus is ordered deliberately for curriculum reasons, at which point shuffling breaks the intent and only a longer run gives coverage",
        falsifier="consumed tokens exceed corpus tokens, meaning the run covered at least one full epoch",
        wrong_claim="The run completed 75 steps as configured, so it trained on the corpus.",
        wrong_why="Step count and coverage are different quantities related by the tokens-per-step product, and at typical settings 75 steps consumes a small fraction of a few-thousand-record corpus.",
        threshold="State intended coverage as a fraction of an epoch and verify it against the token arithmetic before launching.",
        cost="Accelerator hours spent on a run whose coverage cannot support the intended conclusion produce a checkpoint nobody can interpret.",
        scaling="Tokens per step grow with rank count, so the same step count covers more data on a larger job and comparisons across job sizes are invalid.",
        quant=q_token_budget,
    ),
    Mechanism(
        key="effective_batch_hidden", topic="training",
        title="effective batch size is a product of four settings and changes when any of them moves",
        concepts=("batch_size", "learning_rate", "distributed_training"),
        symptom="A learning rate that worked on one cluster produces divergence or no learning on another with identical configuration files.",
        chain="Effective batch is per-device micro-batch times accumulation times rank count, so moving the same configuration to a different rank count changes the effective batch and therefore the appropriate learning rate, without any visible configuration change.",
        metric="Effective batch size computed from all four factors, recorded with every run alongside the learning rate.",
        signature="Runs that differ only in rank count show systematically different loss behaviour at the same nominal learning rate.",
        confounders=(
            "Data ordering differing between clusters, which changes early loss independently.",
            "Mixed precision settings differing, which affects gradient scale.",
            "Warm-up schedule expressed in steps rather than tokens, which changes meaning with batch size.",
        ),
        fixes=(
            "Record effective batch size in the run metadata and reconcile the learning rate against it.",
            "Express warm-up and decay schedules in tokens rather than steps so they survive rank changes.",
            "Fix effective batch by adjusting accumulation when rank count changes.",
        ),
        rollback="Return to the previous rank count and learning rate pairing if loss behaviour changes, rather than tuning the rate against a moving batch.",
        options=("recording effective batch and reconciling the rate against it", "holding effective batch fixed by adjusting accumulation"),
        tradeoff="whether the intent is a fixed effective batch or a fixed per-device batch",
        flip="the goal becomes throughput rather than reproducibility, at which point per-device batch is held and the rate must move with rank count",
        falsifier="loss behaviour is unchanged across rank counts at the same nominal learning rate",
        wrong_claim="The configuration file is identical, so the two runs are the same experiment.",
        wrong_why="Rank count is not in the configuration file but is a factor in effective batch, so identical files on different clusters describe different optimisation problems.",
        threshold="Require effective batch size to be recorded and unchanged before two runs are compared.",
        cost="Runs that are not comparable consume full training cost and cannot be used as evidence for anything.",
        scaling="The discrepancy grows with the ratio of rank counts, so cross-cluster comparison becomes less valid as fleets diverge.",
        quant=q_lr_scale,
    ),
    Mechanism(
        key="prompt_loss_masking", topic="training",
        title="without prompt masking most of the gradient signal optimises reproducing the input",
        concepts=("sft", "loss_masking", "training"),
        symptom="A supervised fine-tune reduces loss steadily and produces a model that echoes prompts rather than answering them.",
        chain="If loss is computed over the whole sequence, the prompt tokens contribute gradient in proportion to their share of the sequence, so a model trained on prompt-heavy records is optimised mainly for predicting prompts.",
        metric="Share of loss-bearing tokens that belong to the answer rather than the prompt.",
        signature="Loss falls faster than answer quality improves, and generated output increasingly mirrors prompt phrasing.",
        confounders=(
            "A learning rate too high, which also degrades output while loss falls.",
            "Corpus repetition allowing memorisation, which lowers loss for a different reason.",
            "Chat template tokens being counted, which shifts the ratio slightly.",
        ),
        fixes=(
            "Mask prompt tokens so only answer tokens contribute loss.",
            "Verify the mask by inspecting the loss-bearing token count for a sample record.",
            "Report the answer share of loss-bearing tokens in the run metadata.",
        ),
        rollback="Revert to the previous checkpoint if masking was absent, since the resulting model was optimised for a different objective and cannot be corrected by further training on the same data.",
        options=("masking prompt tokens so only answers contribute loss", "verifying the mask on sampled records"),
        tradeoff="whether the framework's default applies the mask or computes loss over the full sequence",
        flip="the task genuinely requires modelling the prompt distribution, such as a completion model, at which point full-sequence loss is correct",
        falsifier="the answer share of loss-bearing tokens is already near one",
        wrong_claim="Loss is decreasing smoothly, so the fine-tune is working.",
        wrong_why="Loss decreasing shows the model is fitting whatever tokens carry gradient, and without masking most of those are prompt tokens, so the objective being optimised is not the intended one.",
        threshold="Require the answer share of loss-bearing tokens to be verified on sampled records before a run is accepted.",
        cost="A full training run against the wrong objective produces a checkpoint that must be discarded entirely.",
        scaling="The damage grows with prompt length relative to answer length, so long-context instruction data is affected most.",
        quant=q_masking,
    ),
    Mechanism(
        key="training_memory_multiple", topic="training",
        title="training memory is several times the inference footprint before activations are counted",
        concepts=("training", "optimizer_state", "memory"),
        symptom="A model that serves comfortably on a given device count fails immediately when a fine-tune is attempted on the same hardware.",
        chain="Training holds gradients of the same shape as the weights and optimiser state of several times that size, all resident simultaneously, so the persistent footprint is a multiple of the inference one before any activation memory is allocated.",
        metric="Persistent training footprint computed as weights plus gradients plus optimiser state, per device.",
        signature="The failure occurs during optimiser state allocation rather than during the forward pass, which distinguishes it from an activation problem.",
        confounders=(
            "Activation memory from an over-long sequence, which fails during the forward pass instead.",
            "Gradient checkpointing being enabled or not, which changes the activation term substantially.",
            "Mixed precision holding a separate master copy, which adds another multiple.",
        ),
        fixes=(
            "Compute the persistent footprint from the optimiser in use before scheduling the job.",
            "Shard optimiser state across ranks so no device holds the full copy.",
            "Choose an optimiser with smaller state, accepting its convergence differences.",
        ),
        rollback="Return to the previous optimiser or sharding configuration if convergence degrades, since state reduction trades memory against optimisation behaviour.",
        options=("sharding optimiser state across ranks", "choosing an optimiser with smaller state"),
        tradeoff="whether the memory relief is worth the convergence behaviour it changes",
        flip="sharding communication becomes the bottleneck at the deployed interconnect, at which point a smaller-state optimiser is the better trade",
        falsifier="the failure occurs during the forward pass rather than during state allocation",
        wrong_claim="The model serves fine on these GPUs, so it will fine-tune on them.",
        wrong_why="Serving holds weights and cache while training additionally holds gradients and optimiser state, so the persistent requirement is several times larger before activations are considered.",
        threshold="Verify the computed persistent footprint fits the device before a training job is scheduled.",
        cost="A job that fails at allocation has consumed its scheduling slot and its queue wait and produced nothing.",
        scaling="The multiple is fixed by the optimiser, so the absolute gap grows linearly with parameter count.",
        quant=q_memory_training,
    ),
    Mechanism(
        key="heldout_shares_repetition", topic="training",
        title="a held-out split of a repetitive corpus validates nothing",
        concepts=("validation", "data_quality", "overfitting"),
        symptom="Held-out loss tracks training loss closely throughout the run and the resulting model shows no capability improvement.",
        chain="A split taken from a corpus whose records repeat places the same content in both halves, so the held-out set shares the repetition the model is memorising and its loss falls for the same reason training loss does.",
        metric="Number of distinct questions and distinct answers in each split, and the overlap between them.",
        signature="Both splits contain the same small set of distinct items, and held-out loss falls in lockstep with training loss from the first steps.",
        confounders=(
            "Genuine generalisation, which would also show correlated losses but with capability gains.",
            "A split by row index on an ordered corpus, which separates topics rather than content.",
            "Deduplication applied to one split and not the other.",
        ),
        fixes=(
            "Count distinct content rather than rows before splitting, and split on distinct content.",
            "Hold out whole content families so no family appears in both splits.",
            "Evaluate on a differently sourced set rather than on a split of the same corpus.",
        ),
        rollback="Treat any conclusion drawn from a contaminated split as unsupported, and re-run the evaluation on a differently sourced set before acting on it.",
        options=("splitting on distinct content rather than on rows", "evaluating on a differently sourced set"),
        tradeoff="whether enough distinct content exists to form two genuinely disjoint halves",
        flip="the corpus has too few distinct items to split at all, at which point only an externally sourced evaluation can validate anything",
        falsifier="the two splits share no distinct content and held-out loss diverges from training loss",
        wrong_claim="Held-out loss is falling alongside training loss, so the model is generalising rather than memorising.",
        wrong_why="The held-out set was drawn from the same repetitive corpus, so it contains the same content being memorised and cannot distinguish memorisation from generalisation.",
        threshold="Require the two splits to share no distinct content before held-out loss is treated as evidence.",
        cost="A run validated against a contaminated split can be promoted despite having learned nothing transferable.",
        scaling="The problem worsens as corpus repetition rises, and repetition is invisible in row counts.",
        quant=q_overfit_signal,
    ),
    Mechanism(
        key="untargeted_capability_regression", topic="training",
        title="a narrow fine-tune moves weights shared by capabilities nobody is measuring",
        concepts=("catastrophic_forgetting", "evaluation", "post_training"),
        symptom="A domain fine-tune improves the target task and users report the model became worse at unrelated things.",
        chain="Fine-tuning updates parameters that every capability shares, so gradients that improve the target domain can degrade others, and nothing in the training loss or the domain evaluation observes the degradation.",
        metric="A general-capability probe set evaluated on the same checkpoints as the domain evaluation.",
        signature="Domain score rises while general probe scores fall across the same checkpoint sequence.",
        confounders=(
            "Prompt format changes introduced with the fine-tune, which alter behaviour without weight-level forgetting.",
            "Sampling settings differing between the evaluations.",
            "The general probe being underpowered, so a real regression is not detected.",
        ),
        fixes=(
            "Include a general-capability probe in the acceptance criteria rather than as a follow-up.",
            "Mix general data into the fine-tuning corpus to constrain the update.",
            "Use a parameter-efficient method so the base weights are preserved and the adapter can be removed.",
        ),
        rollback="Revert to the base model or disable the adapter if general capability regresses beyond its stated band, regardless of the domain gain.",
        options=("including a general-capability probe in acceptance", "using a parameter-efficient method that preserves base weights"),
        tradeoff="whether the domain gain requires moving base weights at all",
        flip="the domain gain turns out to require full fine-tuning to reach, at which point the probe becomes the only control and mixing general data is the mitigation",
        falsifier="general probe scores are unchanged across the checkpoint sequence",
        wrong_claim="The fine-tune only saw domain data, so it could only have affected domain behaviour.",
        wrong_why="The data is narrow but the parameters are shared, so gradients from domain data move weights that every other capability also depends on.",
        threshold="Require a general-capability probe with a stated regression band before any fine-tuned checkpoint is promoted.",
        cost="A regression discovered after release costs a rollback plus the trust of everyone who depended on the untargeted capability.",
        scaling="Risk grows as the fine-tuning corpus narrows and as the run length grows, so aggressive domain specialisation is the most exposed case.",
        quant=q_catastrophic,
    ),
    Mechanism(
        key="training_sequence_truncation", topic="training",
        title="a training sequence length below the data length teaches truncated behaviour",
        concepts=("sequence_length", "truncation", "sft"),
        symptom="A fine-tuned model produces answers that stop abruptly before completing their argument.",
        chain="Records longer than the configured sequence length are cut to fit, usually removing the end of the answer, so the model is trained on examples whose answers terminate mid-argument and learns to terminate the same way.",
        metric="Share of training records exceeding the configured sequence length, and the token count discarded from each.",
        signature="The truncation share is material and the model's output length distribution clusters near the training sequence length.",
        confounders=(
            "A generation cap at inference, which truncates output for an unrelated reason.",
            "Stop sequences firing early during generation.",
            "The training data itself containing short answers, which is a data problem rather than a truncation one.",
        ),
        fixes=(
            "Raise the sequence length above the corpus length distribution and re-run.",
            "Filter or split over-length records rather than silently cutting them.",
            "Report the truncation share in run metadata so it is visible before the model is evaluated.",
        ),
        rollback="Discard the checkpoint if the truncation share was material, since continued training on the same data reinforces rather than corrects the behaviour.",
        options=("filtering or splitting over-length records", "raising the sequence length above the data distribution"),
        tradeoff="whether memory permits a sequence length above the corpus distribution",
        flip="memory cannot accommodate the full length, at which point splitting records deliberately is better than cutting them arbitrarily",
        falsifier="the truncation share is near zero",
        wrong_claim="Only a small fraction of records exceed the sequence length, so truncation is not a concern.",
        wrong_why="The truncated records are the long-form answers the fine-tune exists to teach, so a small share by count can be a large share of the intended signal.",
        threshold="Require the truncation share to be reported and to sit below a stated bound before a run is launched.",
        cost="A run that teaches truncated answers must be discarded and repeated, doubling the accelerator cost.",
        scaling="The share rises as answers get longer, so the defect worsens exactly as the task becomes more valuable.",
        quant=q_seq_len_truncation,
    ),
    Mechanism(
        key="data_ordering_reachability", topic="training",
        title="appended data is never reached by a run that stops before the epoch completes",
        concepts=("data_ordering", "curriculum", "training"),
        symptom="A targeted subset was added to the corpus to fix a specific failure and the fine-tuned model shows no change on that failure.",
        chain="Appending records places them at the end of the epoch, so a run configured for a step count below one epoch consumes the earlier portion and terminates before reaching the appended subset at all.",
        metric="Position of the targeted subset within the consumption order, compared against the tokens the run actually consumed.",
        signature="The subset's position lies beyond the run's consumption point, and the number of targeted records seen by the optimiser is zero.",
        confounders=(
            "Shuffling being enabled, which distributes the subset and removes the effect.",
            "The subset being too small to move behaviour even when consumed.",
            "The targeted failure having a different cause that the subset does not address.",
        ),
        fixes=(
            "Interleave the targeted subset across the corpus so a truncated run still reaches it.",
            "Instrument the run to count how many targeted records the optimiser actually consumed.",
            "Extend the run to cover a full epoch if ordering must be preserved.",
        ),
        rollback="Do not conclude the subset is ineffective until the consumption count is non-zero; an unreached subset has not been tested.",
        options=("interleaving the subset across the corpus", "instrumenting the consumed count for the subset"),
        tradeoff="whether the run length can cover the corpus or the ordering must adapt to the run",
        flip="the corpus ordering is deliberate for curriculum reasons, at which point interleaving breaks it and the run must be lengthened instead",
        falsifier="the consumed count for the targeted subset is non-zero",
        wrong_claim="We added the repair data and retrained, and the failure persists, so the repair data does not work.",
        wrong_why="The repair data sat at the end of the corpus and the run stopped before reaching it, so the experiment tested the absence of the data rather than its effect.",
        threshold="Require a non-zero consumed count for any targeted subset before drawing a conclusion about it.",
        cost="A training run that draws the wrong conclusion sends the next round of data work in the wrong direction.",
        scaling="The unreached fraction grows as runs are shortened relative to corpus size, so the problem worsens as corpora grow.",
        quant=q_data_ordering,
    ),
    Mechanism(
        key="authoring_contamination", topic="training",
        title="training data authored from benchmark inspection contaminates that benchmark irreversibly",
        concepts=("contamination", "evaluation", "methodology"),
        symptom="A repair dataset produces a large gain on the benchmark whose failures inspired it and no gain elsewhere.",
        chain="Inspecting a benchmark's failures to decide what the training data should teach transfers information about specific items into the data, so a later evaluation on that benchmark measures the return of that information rather than a generalisable capability.",
        metric="Whether the benchmark's items informed the authoring process, recorded as a process fact rather than inferred from text overlap.",
        signature="The gain concentrates on the inspected benchmark and does not appear on a comparable benchmark the authoring process never saw.",
        confounders=(
            "A genuine capability gain, which would also appear on the uninspected benchmark.",
            "The two benchmarks differing in difficulty or format, which confounds the comparison.",
            "Text-overlap checks passing, which does not address process contamination at all.",
        ),
        fixes=(
            "Reserve a benchmark that the authoring process never sees, and report on it instead.",
            "Record for every dataset which evaluation artifacts informed its creation.",
            "Treat the inspected benchmark as a development set from that point onward, not as evidence.",
        ),
        rollback="Withdraw any capability claim measured on the inspected benchmark; there is no correction that restores it, only a different benchmark.",
        options=("reserving a benchmark the authoring process never sees", "recording which artifacts informed each dataset"),
        tradeoff="whether an uninspected benchmark of comparable quality is available",
        flip="no comparable uninspected benchmark exists, at which point the honest position is that the gain is unmeasured rather than measured and positive",
        falsifier="the gain appears equally on a comparable benchmark the authoring process never saw",
        wrong_claim="We checked for string overlap between the training data and the benchmark and found none, so there is no contamination.",
        wrong_why="Contamination here is a property of the authoring process rather than of the text, so information about the benchmark reached the data through human inspection in a form no string check can detect.",
        threshold="Require a benchmark that demonstrably did not inform the training data before any capability claim is made.",
        cost="A promoted model whose gain is contamination consumes the deployment and the credibility of the next honest result.",
        scaling="Contamination accumulates across rounds, so an organisation that repairs against its benchmark progressively loses the ability to measure anything.",
        quant=q_contamination,
    ),
    Mechanism(
        key="checkpoint_selection_bias", topic="training",
        title="selecting a checkpoint on an evaluation set makes that set part of training",
        concepts=("model_selection", "evaluation", "bias"),
        symptom="The selected checkpoint's reported gain does not reproduce when the model is evaluated on new material.",
        chain="Evaluating several checkpoints on one set and reporting the best conflates the checkpoint's quality with the selection, so the reported figure includes the maximum of several noisy measurements rather than any single unbiased one.",
        metric="Number of candidates compared, reported alongside the selected candidate's score.",
        signature="The gap between the selected checkpoint's score and its score on a fresh set grows with the number of candidates that were compared.",
        confounders=(
            "Genuine differences between checkpoints, which would reproduce on a fresh set.",
            "Evaluation noise being large, which inflates the selection effect.",
            "Checkpoints being correlated, which reduces the effective number of independent comparisons.",
        ),
        fixes=(
            "Select on one set and report on a different, untouched one.",
            "Report the number of candidates compared alongside every selection result.",
            "Pre-register the selection rule, such as the final checkpoint, so no selection occurs.",
        ),
        rollback="Report the pre-registered checkpoint's score rather than the best observed one if no held-out confirmation set is available.",
        options=("selecting on one set and reporting on another", "pre-registering the selection rule"),
        tradeoff="whether enough evaluation material exists to keep a confirmation set untouched",
        flip="evaluation material is too scarce to split, at which point pre-registering the rule is the only remaining control",
        falsifier="the selected checkpoint's score reproduces on a fresh set within the measurement band",
        wrong_claim="We evaluated all three checkpoints and picked the best, which is standard practice.",
        wrong_why="Picking the best of several noisy measurements reports the maximum rather than an unbiased estimate, so the figure is inflated by the selection unless it is confirmed on untouched material.",
        threshold="Require the number of compared candidates to be reported, and confirm the selection on an untouched set.",
        cost="A selection-inflated gain drives a promotion decision that the model cannot sustain in production.",
        scaling="The inflation grows with candidate count, so denser checkpointing makes the bias larger rather than the selection better.",
        quant=q_checkpoint_selection,
    ),
)
