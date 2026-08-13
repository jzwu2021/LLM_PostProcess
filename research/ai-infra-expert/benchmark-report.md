# AI/LLM Infrastructure Domain Expert Benchmark v0.1

日期：2026-08-13
模型：Qwen3.5-9B Base（本阶段未加载、未修改、未评测）
硬件：8 × NVIDIA A30 24GB（本阶段未启动 GPU 训练）

## 1. 阶段目标

本阶段只建立 Domain Expert Benchmark，不进行 Base Model Evaluation，不进行 CPT、SFT、DPO、RLVR 或 Agent 训练。

目标是建立一套能够区分以下能力的固定评估集：

- 事实知识记忆；
- 概念边界理解；
- 公式和容量计算；
- LLM 系统设计；
- 性能瓶颈分析；
- 故障定位；
- 基础代码实现；
- 架构选型；
- 反事实和约束推理；
- 长篇技术分析。

## 2. 数据产物

主文件：

```text
/media/home/johnson/llm/research/ai-infra-expert/benchmark.jsonl
```

生成脚本：

```text
/media/home/johnson/llm/research/ai-infra-expert/build_benchmark.py
```

规模：

```text
500 条
10 个类别
每类 50 条
500 个唯一 ID
500 个唯一问题
```

类别分布：

| Category | Count |
|---|---:|
| Knowledge | 50 |
| Concept Understanding | 50 |
| Calculation | 50 |
| System Design | 50 |
| Performance Analysis | 50 |
| Troubleshooting | 50 |
| Code | 50 |
| Architecture Comparison | 50 |
| Reasoning | 50 |
| Long-form Technical Analysis | 50 |

## 3. Schema

每条记录至少包含用户要求的字段：

```json
{
  "id": "aiinfra-0001",
  "category": "Knowledge",
  "difficulty": "easy|medium|hard",
  "question": "...",
  "reference_answer": "...",
  "verifier": "...",
  "evaluation_method": "..."
}
```

另外增加：

- `topic`：细分知识主题；
- `provenance_status`：当前记录的来源状态；
- `source`：拟使用的官方文档或任务来源；
- `split`：benchmark 版本；
- `contamination_note`：污染控制说明。

## 4. 评估方式

### 4.1 自动验证优先

适合自动验证的类别：

- Calculation：数值解析，允许 1% 相对误差；
- Code：沙箱单元测试；
- 部分 Knowledge/Concept：关键点匹配；
- 结构化配置和公式输出：schema/字段/数值校验。

### 4.2 Rubric 验证

适合 rubric 评分的类别：

- System Design；
- Performance Analysis；
- Troubleshooting；
- Architecture Comparison；
- Reasoning；
- Long-form Technical Analysis。

建议后续使用 1–4 分 rubric，至少包含：

1. 技术正确性；
2. 关键组件或关键点覆盖率；
3. 约束和 trade-off 分析；
4. 验证方法和可操作性；
5. 幻觉、错误因果和不安全建议惩罚。

## 5. 当前题目覆盖

### Knowledge

覆盖 GPU memory hierarchy、HBM/DDR、PCIe/NVLink/NVSwitch、RDMA、RoCE/InfiniBand、CUDA streams、NCCL、Transformer attention、KV cache。

### Concept Understanding

覆盖 prefill/decode、TP/PP、continuous/static batching、DP/EP、quantization/pruning、speculative decoding、MIG/time sharing、CUDA Graphs、GQA/MHA、MoE/dense Transformer。

### Calculation

包含 KV cache 容量、通信传输时间等公式化问题，参数有系统变化，不是同一个数值题的简单复制。

### System / Performance / Troubleshooting

覆盖 70B 多 GPU 服务、RoCE 多节点推理、长上下文、MoE、模型 rollout、Agent inference、GPU capacity planning、分布式训练启动、量化部署和 benchmark harness。

### Code

覆盖 KV cache 计算、TP 合法性、带宽传输时间、重复 request ID、结构化 tool call 解析、paged KV block、prefill/decode 分类、NCCL 环境校验、retry、延迟 percentile。

## 6. 污染控制

当前 benchmark v0.1 是基于审计模板生成的研究 scaffold，不复制 MMLU、GSM8K、HumanEval 或其他公开 benchmark 的题目和答案。

限制：

- 当前题目尚未完成外部专家逐题审核；
- 当前部分开放题使用“关键覆盖点”作为 reference answer，而不是唯一答案；
- `source` 字段目前记录了拟审计来源类型，尚未为每条 factual claim 绑定逐句证据；
- 模板化生成会带来表面模式，不能把 v0.1 直接当作最终发表级 benchmark；
- 尚未完成与 Qwen3.5-9B 预训练语料的污染检测。

下一步应建立 `v0.2`：

1. 对 Knowledge/Concept/Calculation 题目绑定官方文档、论文或 whitepaper 证据；
2. 由领域专家逐题审查；
3. 增加题目重写和对抗变体；
4. 将公开文档中出现过的原句改写为新情境；
5. 创建 private holdout，避免后续训练数据接触；
6. 为开放题建立两个独立 rubric 评分者的一致性测试。

## 7. 阶段验收结果

已通过：

```text
records=500
unique_ids=500
unique_questions=500
missing_required=0
empty_required_fields=0
category_count=10
count_per_category=50
```

未执行：

- Base Model Evaluation；
- GPU 推理；
- 训练；
- benchmark contamination 的语料级检测；
- 专家人工评分。

## 8. 下一阶段建议

下一阶段不直接训练。先完成 benchmark v0.2 质量审计，然后再进行 Base Model Evaluation。

建议的下一阶段最小流程：

1. 抽取 50 条样本做人工技术审查；
2. 修正事实答案、公式和 rubric；
3. 把 500 条拆成公开开发集和私有 holdout；
4. 实现统一 evaluator；
5. 使用原始 Qwen3.5-9B Base 运行完整 benchmark；
6. 记录 accuracy、pass rate、生成 token、总 token、延迟、GPU memory 和 failure type；
7. 汇报 baseline 后等待确认，再决定 CPT 数据构建方案。

本阶段没有对模型进行任何训练或修改。
