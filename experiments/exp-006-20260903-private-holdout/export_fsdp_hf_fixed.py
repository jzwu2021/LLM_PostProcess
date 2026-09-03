"""Export an FSDP checkpoint to HF format.

The shipped /media/home/johnson/llm/scripts/qwen35-9b/export-fsdp-hf.py is broken:
it calls dcp.load into a state dict but never applies it with
model.load_state_dict(), and it calls dcp.load outside the state_dict_type
context. Both of its outputs were byte-identical to each other, i.e. both were
just the base model. This version applies the loaded state inside the context,
matching the loader in exp-005's eval_cross_loss.py that reproduced both
training-time losses exactly.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from transformers import AutoTokenizer

spec = importlib.util.spec_from_file_location(
    "ft", "/media/home/johnson/llm/scripts/qwen35-9b/fine-tune-9b.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-model", default="/media/home/johnson/llm/models/Qwen3.5-9B")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    model = ft.build_model(args.base_model)

    with ft.FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state = {"model": model.state_dict()}
        dcp.load(state, checkpoint_id=args.checkpoint)
        model.load_state_dict(state["model"])

    with ft.FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        full = model.state_dict()

    if rank == 0:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        model.module.save_pretrained(out, state_dict=full, max_shard_size="5GB",
                                     safe_serialization=True)
        AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(out)
        print(f"EXPORTED {args.checkpoint} -> {out}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
