"""Aggregate authored exemplars for the deduplicating stage."""
STANCES = {}

for _module in ("tb_dedup_stances_design", "tb_dedup_stances_code", "tb_dedup_stances_cluster"):
    try:
        STANCES.update(__import__(_module).STANCES)
    except ImportError:
        pass
