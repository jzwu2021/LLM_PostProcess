# Qwen3.5-9B AI/LLM Infrastructure Post-Training Goal

## 1. Research objective

The objective is to adapt Qwen3.5-9B into a reliable AI/LLM Infrastructure engineering assistant. The target is not an unrestricted, autonomous production operator or a claim that the model knows every cluster implementation. The target is a bounded, auditable domain copilot that can explain mechanisms, perform engineering calculations, design systems, analyze performance, diagnose failures, write small verification tools, and use approved tools safely.

The primary hypothesis is:

> At a fixed model size and feasible training-compute budget, high-quality, structured AI/LLM Infrastructure data can improve Qwen3.5-9B on domain knowledge, technical reasoning, system design, performance analysis, troubleshooting, code, and constrained tool-use tasks compared with the untrained Base model.

The corpus currently contains 6,000 authored synthetic seed records, including 1,000 explicit advanced cluster and distributed-inference records. It remains `needs_domain_expert_review`; corpus size alone is not evidence of expert capability.

## 2. Target capability

The intended endpoint is a Level 2–3 engineering copilot, with a controlled exploration of Level 4 tool-mediated operation.

### Level 1: domain question-answering assistant

- Explain common GPU, LLM serving, KV-cache, parallelism, and NCCL concepts.
- State basic assumptions and avoid unsupported platform-specific claims.
- Identify simple trade-offs and common failure modes.

### Level 2: domain engineering copilot

- Perform memory, KV-cache, bandwidth, communication, capacity, and concurrency calculations.
- Design basic multi-GPU serving and training configurations.
- Propose controlled performance experiments.
- Produce small diagnostic scripts and deterministic unit tests.
- Diagnose common OOM, NCCL timeout, KV-cache, and queueing problems.

### Level 3: advanced domain engineering assistant

- Reason about multi-node TP/PP/EP/DP and topology-aware placement.
- Analyze PCIe, NVLink, NVSwitch, RDMA, RoCE, InfiniBand, GDR, and GDS trade-offs.
- Design disaggregated prefill/decode and KV-cache systems, including Mooncake-style architectures and NVIDIA Dynamo-style inference orchestration.
- Produce evidence-driven troubleshooting runbooks with hypotheses, branches, rollback conditions, and required measurements.
- Distinguish model behavior, network behavior, scheduler behavior, and serving-system behavior.

### Level 4: constrained tool-mediated agent

- Select from an approved set of health-check, metrics, topology, NCCL, RDMA, GDS, and serving tools.
- Construct valid tool arguments and use tool results in subsequent reasoning.
- Recover from tool errors or request human confirmation for risky actions.
- Execute only bounded, auditable, reversible operations under explicit permissions and SLOs.

The model is not expected to autonomously operate arbitrary production clusters, make high-risk production changes without approval, or infer unseen hardware behavior without evidence.

## 3. Required task families

### Domain knowledge and concepts

The model should explain GPU memory hierarchy, HBM/DDR, attention, KV cache, prefill/decode, continuous batching, speculative decoding, MoE, quantization, TP/PP/EP/DP, NCCL, PCIe/NVLink/NVSwitch, RDMA/RoCE/InfiniBand, GDR, GDS, Mooncake, NVIDIA Dynamo, scheduling, observability, and failure recovery.

A good answer must include mechanisms, assumptions, applicability, limitations, failure modes, and a validation method rather than only a definition.

### Engineering calculations

The model should calculate or estimate:

- KV-cache memory;
- parameter, activation, gradient, and optimizer memory;
- communication time and effective bandwidth;
- all-reduce/all-to-all cost;
- RDMA and KV-transfer time;
- GPU/NIC oversubscription;
- concurrency and capacity limits;
- checkpoint and GDS I/O time;
- TP/PP/EP degree trade-offs.

Answers must show formulas, units, assumptions, exclusions, and order-of-magnitude checks.

### System design

The model should design:

- single-node multi-GPU serving;
- multi-node TP/PP/EP systems;
- RDMA/RoCE GPU clusters;
- topology-aware GPU/NIC placement;
- prefill/decode disaggregation;
- KV-cache disaggregation and Mooncake-style serving;
- NVIDIA Dynamo-style inference orchestration;
- resilient training with checkpoint recovery;
- gang scheduling, elastic restart, replica routing, autoscaling, and SLO enforcement.

Designs must cover data flow, control flow, topology, resource constraints, observability, failure handling, rollback, and cost/performance trade-offs.

### Performance analysis

The model should form falsifiable hypotheses for issues such as:

- low GPU utilization with saturated memory bandwidth;
- rising P99 latency under higher concurrency;
- RoCE tail-latency instability;
- multi-node NCCL hangs at larger world sizes;
- GDR falling back to host staging;
- GDS not improving checkpoint time;
- speculative decoding being slower than the target model;
- KV disaggregation overhead exceeding its benefit;
- MoE expert imbalance and all-to-all bottlenecks.

Each analysis should specify baselines, workload, metrics, confounders, controlled experiments, expected evidence, and rollback criteria.

### Troubleshooting

The model should produce actionable runbooks for OOM, NCCL initialization and timeout, RDMA queue errors, RoCE congestion, GDR capability failures, GDS stalls, rank-mapping errors, GPU/NIC affinity errors, KV-cache exhaustion, node failure, checkpoint recovery, scheduler fragmentation, and serving SLO violations.

The expected structure is:

```text
symptom -> prioritized hypothesis -> checks -> evidence branches
-> mitigation -> root-cause fix -> regression validation
```

### Code and tool use

The model should write and test small tools for KV-cache estimation, TP validation, NCCL environment checks, RDMA/RoCE checks, GPU/NIC topology parsing, latency percentiles, retry policy, checkpoint capacity, duplicate detection, GDS capability probes, and tool-call JSON validation.

The goal is reliable, bounded engineering automation, not generation of an entire distributed runtime from scratch.

### Agent/tool-call behavior

The model should:

- select the correct diagnostic tool;
- produce valid and complete arguments;
- avoid unnecessary calls;
- use tool results rather than inventing observations;
- handle tool errors and partial results;
- continue multi-turn diagnosis coherently;
- stop or request approval before risky operations;
- distinguish a tool-call syntax pass from an actually useful diagnostic trajectory.

## 4. Training methods and their roles

### CPT

CPT is intended to improve terminology, documentation-style understanding, factual coverage, and concept associations. CPT alone does not establish reliable engineering decision-making or tool-use behavior.

### Structured SFT

SFT is intended to teach the desired behavior and answer structure:

- explicit assumptions;
- formulas and units;
- evidence versus estimates;
- controlled experiments;
- runbook-style diagnosis;
- code with tests;
- native chat-template formatting;
- assistant/tool-call loss masking;
- valid tool-call arguments and multi-turn tool results.

### CPT plus SFT

CPT plus SFT is expected to combine broader domain coverage with executable task behavior. It must be compared against CPT-only and SFT-only controls.

### Preference optimization

Preference optimization is primarily intended to improve correctness, completeness, technical depth, safety, calibration, and hallucination rate. It should not be described as the main source of new factual knowledge.

### RLVR or other verifiable rewards

RLVR is most appropriate for tasks with objective checks:

- numeric calculations;
- capacity and parallelism constraints;
- JSON/tool-call validity;
- code execution and unit tests;
- configuration validation;
- bounded resource-planning tasks.

Its benefit for open-ended architecture design must be measured separately and should not be assumed.

## 5. Evaluation and evidence standard

### Model Domain Capability

This is the primary evidence for domain learning. It must include held-out slices for knowledge, calculation, code, system design, performance analysis, troubleshooting, architecture comparison, reasoning, long-form analysis, and tool-use trajectories.

Required evidence includes:

- Base versus post-trained comparison;
- calculation exact-pass rate;
- code unit-test pass rate;
- rubric scores for design and troubleshooting;
- unseen-parameter and unseen-topology generalization;
- hallucination/error rate;
- tool selection, argument validity, recovery, and final-task success;
- confidence intervals or repeated-run variance;
- contamination checks and private holdout results.

A lower training loss alone is not evidence of domain improvement. A single aggregate accuracy number is also insufficient; multiple independent slices must improve without unacceptable regression.

### Runtime/System Capability

This is kept separate from model-domain learning and measures deployment behavior for a fixed model artifact:

- TTFT, TPOT, and end-to-end latency;
- throughput and concurrency capacity;
- GPU memory peak and headroom;
- KV-cache memory;
- OOM/error rate;
- serving-path tool-call validity and latency;
- recovery time and request loss;
- SLO success;
- GPU-hours per output token.

Runtime metrics cannot be used as evidence that the model learned more AI/LLM Infrastructure knowledge. GPU count, memory, throughput, and topology are execution constraints and system-performance factors, not sources of domain expertise.

## 6. Current status and required gates

Completed:

- deterministic 6,000-record corpus generation;
- 1,000 advanced cluster records;
- benchmark isolation;
- schema, uniqueness, metadata, and manifest validation;
- separate model-domain and runtime-system evaluation specifications;
- Git commit and remote push of the research assets.

Required before a final domain-expertise claim:

1. Expert review of factual, formula, network, and framework-specific records.
2. Evidence links to official documentation, papers, whitepapers, and source code where appropriate.
3. Real logs, topology examples, configuration examples, and failure postmortems.
4. Private holdout and adversarial evaluation sets.
5. A contamination report with frozen hashes.
6. Base, CPT, SFT, CPT+SFT, preference-optimization, and RLVR/tool-use ablations.
7. Parameter-level checkpoint round-trip audit and reproducible inference export.
8. Generation-based tool-use evaluation with actual execution feedback.
9. Separate reporting of Model Domain Capability and Runtime/System Capability.

The final claim should be bounded and evidence-based: the model is an evaluated AI/LLM Infrastructure engineering assistant at measured capability levels, not an unqualified claim of universal cluster expertise.
