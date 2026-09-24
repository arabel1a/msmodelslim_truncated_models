#!/usr/bin/env python3
"""Reproduce the second failure: model.safetensors.index.json does not list
layers.4.hc_attn_fn, while every other layer has it.

Builds the index from the REAL Block parameter names, so the structural
differences between layers (attn.indexer only when compress_ratio==4,
ffn.gate.tid2eid only on hash layers) are present exactly as in a real
checkpoint -- that is what must NOT be mistaken for a missing weight.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.model.common.checkpoint_integrity import ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT
from msmodelslim.model.deepseek_v4.model import Block, ModelArgs
from msmodelslim.model.deepseek_v4.model_adapter import DeepSeekV4ModelAdapter
from msmodelslim.utils.exception import InvalidModelError

NUM_LAYERS = 8
DROP = ("layers.4.hc_attn_fn",)


def build(root: Path, args) -> Path:
    weight_map = {"embed.weight": "shared.safetensors", "head.weight": "shared.safetensors"}
    with torch.device("meta"), patch.object(nn.Linear, "reset_parameters", lambda _s: None):
        for idx in range(NUM_LAYERS):
            for name, _ in Block(idx, args).named_parameters():
                weight_map[f"layers.{idx}.{name}"] = f"layer-{idx}.safetensors"
    for key in DROP:
        weight_map.pop(key, None)

    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name in sorted(set(weight_map.values())):
        (root / name).write_bytes(b"")
    return root


def run(model_path, args_factory, label):
    print(f"\n--- {label} ---")
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    adapter.model_path = model_path
    args = args_factory()
    try:
        adapter.apply_checkpoint_truncation(args)
    except InvalidModelError as err:
        print("raised:", str(err).splitlines()[0])
        print("tip   :", str(err).splitlines()[-1])
        return
    print("num_hidden_layers ->", args.num_hidden_layers)
    print("missing_keys      ->", adapter.checkpoint_integrity.missing_keys)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dsv4-index-"))
    try:
        model_args = ModelArgs()
        model_args.num_hidden_layers = NUM_LAYERS
        model_path = build(tmp / "m", model_args)

        def fresh():
            args = ModelArgs()
            args.num_hidden_layers = NUM_LAYERS
            return args

        os.environ.pop(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, None)
        run(model_path, fresh, "index missing layers.4.hc_attn_fn, no env var")
        os.environ[ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT] = "1"
        run(model_path, fresh, f"same, {ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT}=1")

        # control: a complete index must not be flagged despite per-layer structural
        # differences (indexer on even layers, tid2eid on the first n_hash_layers)
        global DROP
        DROP = ()
        os.environ.pop(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, None)
        run(build(tmp / "complete", model_args), fresh, "complete index (must pass untouched)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
