#!/usr/bin/env python3
"""Reproduce the user's layout (48 shards, only 1-5 and 45-48 on disk) and show
what msmodelslim now does before touching a single weight.

Run:  PYTHONPATH=. python3 tmp_scripts/check_truncated_dsv4_scan.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from msmodelslim.model.common.checkpoint_integrity import (
    ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT,
    scan_checkpoint,
)
from msmodelslim.model.deepseek_v4_dspark.model_adapter import DeepSeekV4DSparkModelAdapter
from msmodelslim.utils.exception import InvalidModelError

NUM_LAYERS = 61
NUM_MTP = 3
NUM_SHARDS = 48
PRESENT = [1, 2, 3, 4, 5, 45, 46, 47, 48]


def shard(i):
    return f"model-{i:05d}-of-{NUM_SHARDS:05d}.safetensors"


def build(root: Path) -> Path:
    """61 layers spread over 48 shards; mtp + head land in the last shards."""
    weight_map = {
        "embed.weight": shard(1),
        "norm.weight": shard(NUM_SHARDS),
        "head.weight": shard(NUM_SHARDS),
    }
    for idx in range(NUM_LAYERS):
        # layers 0..60 -> shards 1..45, roughly the real packing
        file_idx = min(1 + idx * 45 // NUM_LAYERS, 45)
        weight_map[f"layers.{idx}.attn.wkv.weight"] = shard(file_idx)
        weight_map[f"layers.{idx}.ffn.w1.weight"] = shard(file_idx)
    for idx in range(NUM_MTP):
        weight_map[f"mtp.{idx}.attn.wkv.weight"] = shard(45 + idx)
    weight_map["mtp.0.main_proj.weight"] = shard(45)

    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, indent=1))
    for i in PRESENT:
        (root / shard(i)).write_bytes(b"")
    return root


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dsv4-truncated-"))
    try:
        model_path = build(tmp / "DeepSeek-V4-Flash-0731")
        integrity = scan_checkpoint(model_path)
        print("on disk        :", sorted(p.name for p in model_path.glob("*.safetensors"))[:3], "...")
        print("missing shards :", len(integrity.missing_files), "of", integrity.total_files)
        print("usable layers  :", integrity.available_prefix_length("layers"))
        print("usable mtp     :", integrity.available_prefix_length("mtp"))
        print("summary        :", integrity.describe())

        adapter = DeepSeekV4DSparkModelAdapter.__new__(DeepSeekV4DSparkModelAdapter)
        adapter.model_path = model_path

        print("\n--- default (no env var) ---")
        os.environ.pop(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, None)
        args = SimpleNamespace(num_hidden_layers=NUM_LAYERS, n_mtp_layers=NUM_MTP)
        try:
            adapter.apply_checkpoint_truncation(args)
        except InvalidModelError as err:
            print("raised:", err)

        print(f"\n--- with {ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT}=1 ---")
        os.environ[ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT] = "1"
        args = SimpleNamespace(
            num_hidden_layers=NUM_LAYERS,
            n_mtp_layers=NUM_MTP,
            compress_ratios=[1] * NUM_LAYERS,
            dspark_target_layer_ids=(14, 29, 44, 60),
        )
        adapter.apply_checkpoint_truncation(args)
        args.dspark_effective_target_layer_ids = adapter._remap_target_layer_ids(args)
        adapter.config = args
        print("num_hidden_layers ->", args.num_hidden_layers)
        print("n_mtp_layers      ->", args.n_mtp_layers)
        print("target ids        ->", args.dspark_effective_target_layer_ids)
        print("config overrides  ->", adapter.get_truncated_config_overrides())
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
