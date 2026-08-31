"""Core types for the v0.4 AI-infra corpus.

The v0.3 corpus inflated 522 real questions into 5399 rows with a `Scenario
variant N` counter. v0.4 forbids counters: every record must differ by mechanism,
task type or question angle, and every number in an answer is derived from the
deployment setting attached to that record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

GIB = 1024 ** 3


@dataclass(frozen=True)
class Setting:
    """A concrete deployment the question is asked about."""
    key: str
    accel: str
    gpu_count: int
    mem_gb: int
    interconnect: str
    model: str
    params_b: float
    dtype: str
    dtype_bytes: int
    layers: int
    kv_heads: int
    head_dim: int
    hidden: int
    ctx: int
    slo_ms: int
    concurrency: int
    tp: int

    @property
    def label(self) -> str:
        return (f"{self.gpu_count}x {self.accel} {self.mem_gb}GB over {self.interconnect}, "
                f"serving {self.model} ({self.params_b}B, {self.dtype}, TP{self.tp}) "
                f"at context {self.ctx} with a p99 target of {self.slo_ms} ms "
                f"and {self.concurrency} concurrent requests")

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    @property
    def weight_bytes(self) -> int:
        return int(self.params_b * 1e9) * self.dtype_bytes

    @property
    def weight_bytes_per_gpu(self) -> int:
        return self.weight_bytes // self.tp

    @property
    def kv_budget_bytes(self) -> int:
        """Device memory left for KV after weights and a 10% runtime reserve."""
        total = self.mem_gb * GIB
        return int(total * 0.90) - self.weight_bytes_per_gpu


SETTINGS: tuple[Setting, ...] = (
    Setting(
        key="a30-pcie", accel="A30", gpu_count=8, mem_gb=24, interconnect="PCIe only",
        model="Qwen3.5-9B", params_b=9.0, dtype="bf16", dtype_bytes=2,
        layers=48, kv_heads=8, head_dim=128, hidden=4096,
        ctx=8192, slo_ms=2000, concurrency=32, tp=8,
    ),
    Setting(
        key="a100-nvlink", accel="A100", gpu_count=4, mem_gb=80, interconnect="NVLink",
        model="Llama-70B-class", params_b=70.0, dtype="bf16", dtype_bytes=2,
        layers=80, kv_heads=8, head_dim=128, hidden=8192,
        ctx=32768, slo_ms=4000, concurrency=64, tp=4,
    ),
    Setting(
        key="l40s", accel="L40S", gpu_count=2, mem_gb=48, interconnect="PCIe only",
        model="13B-class", params_b=13.0, dtype="fp8", dtype_bytes=1,
        layers=40, kv_heads=8, head_dim=128, hidden=5120,
        ctx=16384, slo_ms=1500, concurrency=24, tp=2,
    ),
    Setting(
        key="h100-ib", accel="H100", gpu_count=16, mem_gb=80, interconnect="NVLink plus InfiniBand",
        model="405B-class", params_b=405.0, dtype="bf16", dtype_bytes=2,
        layers=126, kv_heads=8, head_dim=128, hidden=16384,
        ctx=131072, slo_ms=8000, concurrency=128, tp=8,
    ),
    Setting(
        key="single-4090", accel="RTX 4090", gpu_count=1, mem_gb=24, interconnect="single host",
        model="7B-class", params_b=7.0, dtype="int8", dtype_bytes=1,
        layers=32, kv_heads=8, head_dim=128, hidden=4096,
        ctx=4096, slo_ms=1200, concurrency=8, tp=1,
    ),
)

SETTING_BY_KEY = {s.key: s for s in SETTINGS}


@dataclass(frozen=True)
class Quant:
    """One computed quantity, shown with its derivation."""
    label: str
    steps: Sequence[str]
    value: str
    interpretation: str


@dataclass(frozen=True)
class Mechanism:
    """A specific technical phenomenon, rich enough to answer 30 distinct questions.

    Every text field is authored per mechanism; the generator only supplies
    connective tissue and the derived numbers.
    """
    key: str
    topic: str
    title: str
    concepts: tuple[str, ...]

    symptom: str          # what an operator observes in production
    chain: str            # the causal chain producing that symptom
    metric: str           # the measurement that discriminates this cause
    signature: str        # the observation that confirms it over rivals
    confounders: tuple[str, str, str]
    fixes: tuple[str, str, str]        # ordered cheapest and most reversible first
    rollback: str
    options: tuple[str, str]           # two competing approaches; [0] is the cheaper, reversible one
    tradeoff: str                      # what actually decides between them
    flip: str                          # the condition under which options[0] stops being right
    falsifier: str                     # observation that would refute the hypothesis
    wrong_claim: str                   # plausible but wrong statement
    wrong_why: str
    threshold: str                     # an operational threshold rule
    cost: str                          # efficiency or spend angle
    scaling: str                       # behaviour as scale or concurrency grows
    quant: Callable[[Setting], Quant]

    def concept_list(self) -> list[str]:
        return list(self.concepts)


REGISTRY: list[Mechanism] = []


SENTENCE_FIELDS = ("symptom", "chain", "metric", "signature", "rollback",
                   "wrong_claim", "wrong_why", "threshold", "cost", "scaling")
PHRASE_FIELDS = ("title", "tradeoff", "flip", "falsifier")


def _check_sentence(where: str, text: str) -> None:
    if not text or not text[0].isupper():
        raise ValueError(f"{where}: sentence field must start with a capital: {text[:60]!r}")
    if not text.rstrip().endswith("."):
        raise ValueError(f"{where}: sentence field must end with a period: {text[-60:]!r}")


def _check_phrase(where: str, text: str) -> None:
    first = (text or " ").split(" ", 1)[0].strip("(,")
    # acronyms such as KV, RDMA, NCCL are legitimately capitalised mid-sentence
    if not text or (first[:1].isupper() and not first.isupper()):
        raise ValueError(f"{where}: phrase field must start lowercase: {text[:60]!r}")
    if text.rstrip().endswith("."):
        raise ValueError(f"{where}: phrase field must not end with a period: {text[-60:]!r}")


def validate(m: Mechanism) -> None:
    """Composers splice these into fixed grammatical slots, so the shape is part of the contract."""
    for f in SENTENCE_FIELDS:
        _check_sentence(f"{m.key}.{f}", getattr(m, f))
    for f in PHRASE_FIELDS:
        _check_phrase(f"{m.key}.{f}", getattr(m, f))
    for i, c in enumerate(m.confounders):
        _check_sentence(f"{m.key}.confounders[{i}]", c)
    for i, x in enumerate(m.fixes):
        _check_sentence(f"{m.key}.fixes[{i}]", x)
    for i, o in enumerate(m.options):
        _check_phrase(f"{m.key}.options[{i}]", o)


def _tokens(text: str) -> set:
    return {w for w in text.lower().replace(".", " ").split() if len(w) > 3}


def _check_distinct(m: Mechanism) -> None:
    """Two mechanisms restating the same idea is the failure v0.3 made; catch it at authoring time."""
    for f in ("title", "threshold", "chain"):
        new = _tokens(getattr(m, f))
        for prior in REGISTRY:
            old = _tokens(getattr(prior, f))
            if not new or not old:
                continue
            overlap = len(new & old) / len(new | old)
            if overlap > 0.5:
                raise ValueError(
                    f"{m.key}.{f} overlaps {prior.key}.{f} at {overlap:.2f}; "
                    f"these are the same mechanism stated twice")


def register(*mechs: Mechanism) -> None:
    seen = {m.key for m in REGISTRY}
    for m in mechs:
        if m.key in seen:
            raise ValueError(f"duplicate mechanism key {m.key}")
        validate(m)
        _check_distinct(m)
        REGISTRY.append(m)
        seen.add(m.key)


def gib(n_bytes: float) -> str:
    return f"{n_bytes / GIB:.2f} GiB"


def fmt_int(n: float) -> str:
    return f"{int(n):,}"
