"""Symmetric cross-evaluation of base, exp-002 and exp-004 checkpoints.

There is no unbiased evaluation set available for this comparison:

  * exp-002's held-out set is native to exp-002's corpus style;
  * exp-004's held-out set is native to exp-004's corpus style;
  * benchmark.jsonl is contaminated for exp-002, whose repair records were
    authored after inspecting benchmark regression topics.

So every arm is run on every set and the whole matrix is reported with the bias
of each cell declared. A model winning on its own home set is a sanity check, not
evidence. The informative cells are off-diagonal.

Masked loss is style-dependent: it partly measures "does this target text look
like my training data". It is reported as a fit measure, never as capability.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import StateDictType
from transformers import AutoTokenizer

spec = importlib.util.spec_from_file_location(
    "ft", "/media/home/johnson/llm/scripts/qwen35-9b/fine-tune-9b.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

BASE = "/media/home/johnson/llm/models/Qwen3.5-9B"


def evaluate(model, tokenizer, records, seq_len, local_rank):
    """Returns the mean and the per-item losses; per-item is needed for a paired test."""
    model.eval()
    per_item = []
    with torch.no_grad():
        for record in records:
            input_ids, attn, labels, targets, _ = ft.encode_masked(tokenizer, record, seq_len)
            if targets == 0:
                continue
            out = model(input_ids=input_ids.to(local_rank),
                        attention_mask=attn.to(local_rank),
                        labels=labels.to(local_rank), use_cache=False)
            per_item.append((record.get("id"), float(out.loss.detach().float().item())))
    model.train()
    mean = sum(v for _, v in per_item) / max(len(per_item), 1)
    return mean, per_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="FSDP checkpoint dir; omit for the base model")
    ap.add_argument("--label", required=True)
    ap.add_argument("--datasets", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--examples", type=int, default=0, help="0 evaluates the whole file")
    ap.add_argument("--seq-len", type=int, default=768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = ft.build_model(BASE)

    if args.checkpoint:
        with ft.FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state = {"model": model.state_dict()}
            dcp.load(state, checkpoint_id=args.checkpoint)
            model.load_state_dict(state["model"])
        if rank == 0:
            print(f"LOADED {args.checkpoint}", flush=True)
    elif rank == 0:
        print("LOADED base model (no checkpoint)", flush=True)

    results = {}
    for pair in args.datasets:
        name, path = pair.split("=", 1)
        records = ft.load_records(path)
        if args.examples:
            records = records[: args.examples]
        loss, per_item = evaluate(model, tokenizer, records, args.seq_len, local_rank)
        results[name] = {"loss": loss, "examples": len(records), "path": path,
                         "per_item": per_item}
        if rank == 0:
            print(f"EVAL model={args.label} set={name} loss={loss:.6f} n={len(records)}", flush=True)

    if rank == 0:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"model": args.label, "checkpoint": args.checkpoint,
             "seq_len": args.seq_len, "results": results}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
