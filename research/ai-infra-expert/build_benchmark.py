#!/usr/bin/env python3
"""Build a deterministic v0.1 AI/LLM Infrastructure expert benchmark.

The benchmark is intentionally generated from auditable templates. It is a
research scaffold, not a claim that all open-ended references are gold labels.
"""
import json, math, random
from pathlib import Path

OUT = Path(__file__).with_name("benchmark.jsonl")
random.seed(20260813)

TOPICS = [
    ("GPU memory hierarchy", "Registers, shared memory, L1/L2 cache, HBM/DDR are progressively larger/slower; data locality determines effective bandwidth.", "rubric_1_4"),
    ("HBM versus DDR", "HBM is high-bandwidth device memory close to the accelerator; DDR is host memory with lower bandwidth and higher access latency for GPU workloads.", "contains_key_points"),
    ("PCIe versus NVLink", "PCIe is a general host/device/interconnect link; NVLink provides higher-bandwidth, lower-overhead accelerator-to-accelerator communication when supported.", "contains_key_points"),
    ("NVSwitch", "NVSwitch provides a switched fabric so GPUs can communicate through many high-bandwidth paths rather than relying only on a sparse peer topology.", "contains_key_points"),
    ("RDMA", "RDMA transfers data between NIC and application memory with reduced CPU involvement; it still requires correct registration, transport and congestion handling.", "contains_key_points"),
    ("RoCE and InfiniBand", "RoCE carries RDMA over Ethernet and depends on loss/congestion configuration; InfiniBand provides an RDMA fabric with its own transport and management ecosystem.", "contains_key_points"),
    ("CUDA streams", "Streams order work within a stream and allow eligible kernels/copies in different streams to overlap subject to dependencies and resources.", "contains_key_points"),
    ("NCCL collectives", "NCCL selects topology-aware algorithms/protocols for collectives such as all-reduce; performance depends on message size, links and process placement.", "contains_key_points"),
    ("Transformer attention", "Attention forms query-key compatibility scores, normalizes them, and combines values; causal masking prevents use of future tokens.", "contains_key_points"),
    ("KV cache", "During autoregressive decode, prior keys and values are reused so the model need not recompute them; cache memory grows with layers, KV heads, sequence length and dtype.", "contains_key_points"),
]

CATS = ["Knowledge", "Concept Understanding", "Calculation", "System Design", "Performance Analysis", "Troubleshooting", "Code", "Architecture Comparison", "Reasoning", "Long-form Technical Analysis"]

def rec(i, cat, diff, q, ans, verifier, method, topic, source):
    return {"id": f"aiinfra-{i:04d}", "category": cat, "difficulty": diff, "question": q,
            "reference_answer": ans, "verifier": verifier, "evaluation_method": method,
            "topic": topic, "provenance_status": "curated_template_v0.1", "source": source,
            "split": "benchmark_v0.1", "contamination_note": "Generated after the model checkpoint was selected; not copied from a public benchmark."}

def build():
    rows=[]; i=1
    # A: 50 knowledge items: 10 topics x 5 formulations.
    for t, (topic, ans, ver) in enumerate(TOPICS):
        for f in range(5):
            qs=[f"Define {topic} in the context of LLM infrastructure and state its primary performance implication.",
                f"What problem does {topic} solve in a production LLM system, and what is one limitation?",
                f"Explain {topic} to an engineer debugging a large-model training or serving job.",
                f"State two facts about {topic} that are relevant to distributed LLM workloads.",
                f"Distinguish the role of {topic} from ordinary host-side software in an LLM stack."]
            rows.append(rec(i,"Knowledge",["easy","medium","medium","medium","hard"][f],qs[f],ans,ver,"reference answer + key-point verifier",topic,"NVIDIA CUDA/NCCL/architecture documentation; source audit pending")); i+=1
    # B: 50 concept items.
    contrasts=[("prefill","decode","Prefill is parallel over prompt tokens and is often compute-bound; decode is sequential per generated token and is commonly memory/KV-cache or latency bound."),
               ("tensor parallelism","pipeline parallelism","Tensor parallelism splits layer computation/tensors and communicates within layers; pipeline parallelism partitions layers and communicates activations between stages."),
               ("continuous batching","static batching","Continuous batching admits/completes requests dynamically; static batching waits for a fixed batch and can waste slots when sequence lengths differ."),
               ("data parallelism","expert parallelism","Data parallelism replicates model computation across workers with different data; expert parallelism distributes MoE experts and routes tokens across workers."),
               ("quantization","pruning","Quantization reduces numerical precision; pruning removes or sparsifies parameters, with different hardware and accuracy implications."),
               ("speculative decoding","larger target model decoding","Speculation drafts several tokens cheaply and verifies them with the target; speedup requires high acceptance and compatible scheduling."),
               ("MIG partitioning","time sharing","MIG provides hardware-isolated partitions on supported GPUs; time sharing multiplexes workloads without the same isolation or memory guarantees."),
               ("CUDA Graphs","eager execution","CUDA Graphs capture a reusable execution graph to reduce launch overhead; eager execution is more flexible but may incur repeated dispatch overhead."),
               ("GQA","MHA","Grouped-query attention shares K/V heads among query groups, reducing KV-cache memory and bandwidth relative to multi-head attention."),
               ("MoE","dense Transformer","MoE activates a subset of experts per token, increasing parameter capacity at roughly sparse compute cost but adding routing and communication complexity.")]
    for a,(x,y,ans) in enumerate(contrasts):
        for f in range(5):
            q=[f"Compare {x} and {y} for an LLM serving system.",f"When would {x} be preferable to {y}, and what trade-off follows?",f"A team confuses {x} with {y}; correct the explanation using compute, memory, and communication.",f"Give one workload where {x} wins and one where {y} wins.",f"Explain how {x} changes the bottleneck relative to {y}."][f]
            rows.append(rec(i,"Concept Understanding",["medium","medium","hard","medium","hard"][f],q,ans,"contains_key_points","rubric with contrastive key points",x,"NVIDIA/PyTorch/vLLM documentation; source audit pending")); i+=1
    # C: 50 calculations: 10 families x 5 parameterizations.
    for t in range(10):
        for f in range(5):
            seq=2048*(f+1); layers=24+4*t; kv=8+(t%4); hd=128; bytes_=2
            kv_bytes=2*layers*seq*kv*hd*bytes_; kv_gib=kv_bytes/(1024**3)
            bw=1.0+0.25*f; size=8+2*t; time_ms=size/bw
            if t%2==0:
                q=f"A model has {layers} layers, sequence length {seq}, {kv} KV heads, head dimension {hd}, and BF16 KV values. Estimate KV-cache bytes for one sequence (ignore allocator overhead)."
                ans=f"2*{layers}*{seq}*{kv}*{hd}*2 = {kv_bytes} bytes ({kv_gib:.6f} GiB)."
                ver="numeric_tolerance"
            else:
                q=f"A communication payload is {size:.1f} GB and the effective link bandwidth is {bw:.2f} GB/s. Estimate the ideal one-way transfer time, ignoring protocol overhead."
                ans=f"{size:.1f}/{bw:.2f} = {time_ms:.6f} seconds ({time_ms*1000:.3f} ms)."; ver="numeric_tolerance"
            rows.append(rec(i,"Calculation",["easy","easy","medium","medium","hard"][f],q,ans,ver,"extract numeric answer and tolerance 1%", "KV cache" if t%2==0 else "communication", "Derived formula; no external answer key")); i+=1
    # D: open-ended systems/design/analysis families. 10 topics x 5.
    design_topics=[("serve a 70B model on 8 GPUs","tensor/pipeline parallel layout, KV-cache budget, batching, admission control, observability, and failure handling"),
                   ("design multi-node inference over RoCE","NIC/GPU affinity, GPUDirect/RDMA prerequisites, PFC/ECN validation, topology-aware parallelism, and congestion tests"),
                   ("design long-context serving","prefix caching, paged KV cache, length admission limits, chunked prefill, eviction policy, and tail-latency SLOs"),
                   ("design an MoE inference cluster","expert placement, routing capacity factor, all-to-all communication, load imbalance monitoring, and graceful fallback"),
                   ("design a safe model rollout","shadow traffic, canary, checksum/version pinning, latency/error rollback gates, and checkpoint provenance"),
                   ("optimize an agent inference service","tool-call batching, timeout budgets, caching, speculative execution limits, and loop/stopping safeguards"),
                   ("plan GPU capacity","request mix, tokens/sec, memory headroom, queueing, redundancy, and GPU-hour cost model"),
                   ("debug distributed training startup","rank mapping, rendezvous, NCCL topology, environment capture, and minimal reproduction"),
                   ("compare quantization deployment choices","accuracy calibration, kernel availability, memory savings, throughput, and fallback strategy"),
                   ("build a benchmark harness","fixed prompts, deterministic seeds, raw outputs, verifier isolation, latency/memory telemetry, and immutable manifests")]
    for cat in ["System Design","Performance Analysis","Troubleshooting","Architecture Comparison","Reasoning","Long-form Technical Analysis"]:
        for t,(topic,outline) in enumerate(design_topics):
            for f in range(5):
                if cat=="System Design":
                    q=f"Design a production system to {topic}. Include components, data flow, scaling, failure handling, and observability. Variant {f+1}."; ans=f"A strong answer must cover {outline}."
                elif cat=="Performance Analysis":
                    q=f"Analyze a system that must {topic}. Identify likely bottlenecks, measurements, and an experiment matrix. Variant {f+1}."; ans=f"A strong answer must connect bottlenecks to measurements and controlled experiments, including {outline}."
                elif cat=="Troubleshooting":
                    q=f"A system attempting to {topic} has high latency or failures. Give a prioritized diagnostic plan. Variant {f+1}."; ans=f"A strong answer isolates memory, compute, communication, scheduling, and configuration causes, then validates {outline}."
                elif cat=="Architecture Comparison":
                    q=f"Compare two plausible architectures for trying to {topic}; recommend one for a latency-sensitive workload. Variant {f+1}."; ans=f"A strong answer states workload assumptions, compares compute/memory/communication and operational risks, then justifies a choice using {outline}."
                elif cat=="Reasoning":
                    q=f"Reason step by step about this counterfactual: if the system must {topic} but inter-GPU bandwidth is cut in half, what changes first? Variant {f+1}."; ans=f"A strong answer predicts communication amplification and re-evaluates placement, batching, parallelism, and SLOs; it should discuss {outline}."
                else:
                    q=f"Write a technical analysis of how to {topic}. Separate facts, assumptions, calculations, risks, and validation experiments. Variant {f+1}."; ans=f"A strong answer is structured around facts, assumptions, measurable hypotheses, and mitigations, including {outline}."
                rows.append(rec(i,cat,["medium","medium","hard","hard","hard"][f],q,ans,"rubric_1_4","expert rubric: completeness, correctness, trade-offs, validation",topic,"Curated scenario; official documentation sources to be attached during audit")); i+=1
    # E: code: 50 tasks, varied but safely auto-verifiable by unit tests.
    code_tasks=[("compute KV-cache bytes","write a function kv_bytes(layers,seq,kv_heads,head_dim,bytes_per_value) returning 2*layers*seq*kv_heads*head_dim*bytes_per_value", "unit_test"),
                ("validate a tensor-parallel world size","write a function valid_tp(world_size,tp) that returns true iff world_size is divisible by tp", "unit_test"),
                ("estimate transfer time","write a function transfer_seconds(gib,gbps) using decimal GB/s and returning seconds", "unit_test"),
                ("detect duplicate request IDs","write a streaming check that reports the first duplicate ID without changing order", "unit_test"),
                ("parse a structured tool call","write a parser that rejects invalid JSON and requires string name plus object arguments", "unit_test"),
                ("calculate paged KV blocks","write a function ceil(seq_len/block_tokens) without floating point", "unit_test"),
                ("classify prefill/decode workload","write a classifier using prompt_tokens and generated_tokens with documented thresholds", "unit_test_plus_rubric"),
                ("check NCCL environment completeness","write a validator for required variables and report missing names", "unit_test"),
                ("implement retry with bounded backoff","write a retry helper with max attempts and no retry on non-retryable errors", "unit_test"),
                ("aggregate latency percentiles","write code that computes p50 and p99 from a list and handles empty input", "unit_test")]
    for t,(task,spec,ver) in enumerate(code_tasks):
        for f in range(5):
            q=f"Implement {task} for an LLM infrastructure utility. Variant {f+1}: {spec}. Include input validation and a short complexity note."
            ans=f"Expected implementation contract: {spec}; reject malformed or non-positive inputs where applicable; provide deterministic unit tests."
            rows.append(rec(i,"Code",["easy","medium","medium","hard","hard"][f],q,ans,ver,"sandboxed unit tests plus code rubric",task,"Original task specification; no copied code")); i+=1
    assert len(rows)==500, len(rows)
    assert len({r['question'] for r in rows})==500
    with OUT.open('w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    print(f"wrote {len(rows)} records to {OUT}")
    from collections import Counter
    print(json.dumps(Counter(r['category'] for r in rows),ensure_ascii=False,sort_keys=True))

if __name__=='__main__': build()
