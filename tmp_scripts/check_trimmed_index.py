#!/usr/bin/env python3
"""The -bf16 layout: the conversion kept shards 1-5 and 45-48 of 48 AND rewrote
model.safetensors.index.json to list only what it wrote.

So nothing is "missing": every file the index references exists. The index just
stops at layers.3 while config.json still declares 43 layers. That is what has to
be detected, and what msmodelslim used to sail straight past.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.model.common.checkpoint_integrity import ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, scan_checkpoint
from msmodelslim.model.deepseek_v4.model import Block, ModelArgs
from msmodelslim.model.deepseek_v4.model_adapter import DeepSeekV4ModelAdapter
from msmodelslim.utils.exception import InvalidModelError

DECLARED_LAYERS = 43
KEPT_LAYERS = 4
NUM_MTP = 3


def args_for(num_layers=DECLARED_LAYERS):
    args = ModelArgs()
    args.num_hidden_layers = num_layers
    args.n_mtp_layers = NUM_MTP
    args.n_routed_experts = 2
    args.max_seq_len = 512
    return args


def build(root: Path) -> Path:
    args = args_for()
    weight_map = {
        "embed.weight": "model-00001-of-00048.safetensors",
        "norm.weight": "model-00045-of-00048.safetensors",
        "head.weight": "model-00045-of-00048.safetensors",
    }
    with torch.device("meta"), patch.object(nn.Linear, "reset_parameters", lambda _s: None):
        for idx in range(KEPT_LAYERS):
            for name, _ in Block(idx, args).named_parameters():
                weight_map[f"layers.{idx}.{name}"] = f"model-{idx + 2:05d}-of-00048.safetensors"
    for idx in range(NUM_MTP):
        weight_map[f"mtp.{idx}.attn.wkv.weight"] = f"model-{45 + idx:05d}-of-00048.safetensors"

    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name in sorted(set(weight_map.values())):
        (root / name).write_bytes(b"")
    return root


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dsv4-trimmed-"))
    try:
        model_path = build(tmp / "DeepSeek-V4-Flash-0731-bf16")
        integrity = scan_checkpoint(model_path)
        print("files referenced by the index that are absent :", integrity.missing_files)
        print("blocks the index covers                       :", integrity.available_prefix_length("layers"))
        print("config.json declares                          :", DECLARED_LAYERS)

        for env, label in ((None, "no env var"), ("1", f"{ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT}=1")):
            print(f"\n--- {label} ---")
            if env is None:
                os.environ.pop(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, None)
            else:
                os.environ[ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT] = env
            adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
            adapter.model_path = model_path
            args = args_for()
            try:
                adapter.apply_checkpoint_truncation(args)
            except InvalidModelError as err:
                print("raised:", str(err).splitlines()[0])
                continue
            print("num_hidden_layers ->", args.num_hidden_layers)
            print("n_mtp_layers      ->", args.n_mtp_layers)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
