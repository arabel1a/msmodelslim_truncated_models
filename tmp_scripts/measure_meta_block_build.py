#!/usr/bin/env python3
"""How long does it take to build DeepSeek-V4 Blocks on the meta device?

If cheap, the "which parameters does layer N need" question can be answered
exactly instead of guessed at from the index's key patterns.
"""

import time
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.model.deepseek_v4.model import Block, ModelArgs

args = ModelArgs()
print("n_routed_experts", args.n_routed_experts, "n_hash_layers", args.n_hash_layers)
print("compress_ratios[:8]", list(args.compress_ratios[:8]))

torch.set_default_dtype(torch.bfloat16)
for layer_id in (0, 2, 3):
    start = time.perf_counter()
    with torch.device("meta"), patch.object(nn.Linear, "reset_parameters", lambda _s: None):
        block = Block(layer_id, args)
    names = {name for name, _ in block.named_parameters()}
    elapsed = time.perf_counter() - start
    has = lambda p: any(n.startswith(p) for n in names)  # noqa: E731
    print(
        f"layer {layer_id}: {elapsed:.3f}s, {len(names)} params, "
        f"ratio={args.compress_ratios[layer_id]}, compressor={has('attn.compressor')}, "
        f"indexer={has('attn.indexer')}, tid2eid={'ffn.gate.tid2eid' in names}, "
        f"gate_bias={'ffn.gate.bias' in names}, hc_attn_fn={'hc_attn_fn' in names}"
    )
