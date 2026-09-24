#!/usr/bin/env python3
"""Does building blocks under torch.device('meta') poison the model's cached
tensor factories for everything built afterwards?

Attention registers freqs_cis from precompute_freqs_cis(), which is @lru_cache(2)
and allocates on the CURRENT default device.
"""

from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.model.deepseek_v4 import model as v4_model
from msmodelslim.model.deepseek_v4.model import Attention, ModelArgs


def small_args():
    args = ModelArgs()
    args.num_hidden_layers = 4
    args.n_routed_experts = 2
    args.max_seq_len = 512
    return args


def report(label, module):
    print(f"{label:<44} freqs_cis.is_meta={module.freqs_cis.is_meta}  kv_cache.is_meta={module.kv_cache.is_meta}")


args = small_args()
torch.set_default_dtype(torch.bfloat16)

for fn in (v4_model.precompute_freqs_cis, v4_model.get_window_topk_idxs, v4_model.get_compress_topk_idxs):
    fn.cache_clear()

report("baseline: built on cpu, no meta involved", Attention(0, args))

for fn in (v4_model.precompute_freqs_cis,):
    fn.cache_clear()

with torch.device("meta"), patch.object(nn.Linear, "reset_parameters", lambda _s: None):
    Attention(0, args)
report("after a meta build: next cpu build", Attention(0, args))

v4_model.precompute_freqs_cis.cache_clear()
report("after cache_clear(): next cpu build", Attention(0, args))
