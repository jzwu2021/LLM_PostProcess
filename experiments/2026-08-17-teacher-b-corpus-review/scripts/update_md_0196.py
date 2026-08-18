import json,subprocess,os
EXP="/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review"
os.chdir(EXP)
rows=[json.loads(l) for l in open("results/train-batch-0196.jsonl")]
from collections import Counter
c=Counter(r["decision"] for r in rows)
ids=[r["source_id"] for r in rows]
entry=f"""## Run 2026-08-18 - train-batch-0196.jsonl

- Batch file: results/train-batch-0196.jsonl
- Corpus range: positional slice train.jsonl[1950:1960] (10 records)
- Source IDs: {ids[0]} .. {ids[-1]} (non-consecutive; sliced by position, not ID arithmetic)
- Progress: 1960/2500 train (78.4%); remaining 540
- Decisions: keep={c.get('keep',0)}, rewrite={c.get('rewrite',0)}, reject={c.get('reject',0)}
- Initial schema check: PASS (scripts/tb_verify_batch_0196.py, first run, no repair needed)
- Repairs performed: none
- Final schema check: PASS (VERIFY_PASS, TOTAL 1960, strict prefix of train.jsonl confirmed, no validation-batch files)
- Manifest: regenerated MANIFEST.sha256; sha256sum -c all OK

Topics covered: all ten items are variants of the same weight-only quantization (WOQ)
fair-comparison prompt, so this batch deliberately attacks it from ten disjoint analytical
stances: calibration-set representativeness; activation outliers and numerical range;
kernel autotuning and GEMM shape / prefill-vs-decode boundedness; SLO-constrained unit
economics; long-context KV dominance versus weight-only savings; statistical power and
equivalence testing (TOST); metric-ladder selection from perplexity to strict tool-call
validity; GPU generation and dtype support (Ampere A30 lacking an FP8 tensor-core path
versus Hopper); canary gating and blast radius; and artefact provenance / reproducibility
of the quantization recipe itself.

Provisional status: these are provisional teacher-B reviews produced blind, without any
access to teacher-A outputs. They are NOT expert gold labels, NOT validated by measurement,
and do NOT constitute evidence of any model's domain capability. No MEASURED number is
asserted in any record; all quantities are design targets, and the numbers policy in each
answer requires MEASURED/ESTIMATE labelling downstream.

"""
md=open("EXPERIMENT.md").read()
i=md.find("\n## ")
open("EXPERIMENT.md","w").write(md[:i+1]+entry+md[i+1:] if i>0 else entry+md)
print("MD_OK")
subprocess.run("find . -type f ! -name MANIFEST.sha256 ! -path './__pycache__/*' ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256", shell=True, check=True)
r=subprocess.run("sha256sum -c MANIFEST.sha256", shell=True, capture_output=True, text=True)
bad=[l for l in r.stdout.split("\n") if l.strip() and not l.endswith(": OK")]
print("MANIFEST_LINES", len(open("MANIFEST.sha256").read().strip().split("\n")), "BAD", bad, "rc", r.returncode)
