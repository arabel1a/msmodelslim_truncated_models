#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2
-------------------------------------------------------------------------

截断权重下 DSpark 适配器对 dspark_target_layer_ids 的重定位。

mtp.0.main_proj 的入维是 dim * len(dspark_target_layer_ids)，所以 target 的**个数**不能变；
被截掉的层号统一压到最后一层，前向按出现次数各取一份 hidden。
"""

from types import SimpleNamespace

import torch
from torch import nn

from msmodelslim.model.deepseek_v4_dspark.model_adapter import DeepSeekV4DSparkModelAdapter

DIM = 8


def make_adapter(config):
    adapter = DeepSeekV4DSparkModelAdapter.__new__(DeepSeekV4DSparkModelAdapter)
    adapter.config = config
    return adapter


class DummyBlock(nn.Module):
    def forward(self, h, start_pos, input_ids):  # pragma: no cover - 只为触发前置 hook
        return h


class DummyModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(DummyBlock() for _ in range(num_layers))

    def forward(self, input_ids):
        h = torch.zeros(1, 4, 2, DIM)
        return self.layers[0](h, 0, input_ids)


class TestRemapTargetLayerIds:
    def test_ids_within_range_are_untouched(self):
        args = SimpleNamespace(dspark_target_layer_ids=(0, 1, 2), num_hidden_layers=3)

        assert DeepSeekV4DSparkModelAdapter._remap_target_layer_ids(args) == (0, 1, 2)

    def test_out_of_range_ids_are_clamped_to_the_last_layer(self):
        args = SimpleNamespace(dspark_target_layer_ids=(14, 29, 44, 60), num_hidden_layers=3)

        assert DeepSeekV4DSparkModelAdapter._remap_target_layer_ids(args) == (2, 2, 2, 2)

    def test_the_number_of_targets_is_preserved(self):
        args = SimpleNamespace(dspark_target_layer_ids=(1, 14, 29), num_hidden_layers=3)

        remapped = DeepSeekV4DSparkModelAdapter._remap_target_layer_ids(args)

        assert len(remapped) == 3
        assert remapped == (1, 2, 2)

    def test_no_targets_declared(self):
        args = SimpleNamespace(dspark_target_layer_ids=(), num_hidden_layers=3)

        assert DeepSeekV4DSparkModelAdapter._remap_target_layer_ids(args) == ()


class TestTruncatedConfigOverrides:
    def test_remapped_targets_are_written_back(self):
        adapter = make_adapter(
            SimpleNamespace(
                checkpoint_truncated=True,
                num_hidden_layers=3,
                n_mtp_layers=2,
                compress_ratios=[1] * 6,
                dspark_target_layer_ids=(14, 29, 44),
                dspark_effective_target_layer_ids=(2, 2, 2),
            )
        )

        overrides = adapter.get_truncated_config_overrides()

        assert overrides["dspark_target_layer_ids"] == [2, 2, 2]
        assert overrides["num_hidden_layers"] == 3

    def test_nothing_is_written_for_a_complete_checkpoint(self):
        adapter = make_adapter(
            SimpleNamespace(
                num_hidden_layers=61,
                dspark_target_layer_ids=(14, 29, 44),
                dspark_effective_target_layer_ids=(14, 29, 44),
            )
        )

        assert adapter.get_truncated_config_overrides() == {}


class TestMainHiddenWidth:
    """前向采到的 main_hidden 宽度必须始终等于 dim * len(target_ids)。"""

    @staticmethod
    def _run_until_mtp(adapter, num_layers):
        model = DummyModel(num_layers)
        captured = {}

        def fake_preprocess(model, mtp_decoder, mtp_idx, main_hidden, input_ids, start_pos, kwargs):
            captured["main_hidden"] = main_hidden
            return (torch.zeros(1, 1, DIM), start_pos, input_ids), {**kwargs, "main_x": torch.zeros(1, 1, DIM)}

        adapter.dspark_mtp_preprocess = fake_preprocess
        adapter.generate_decoder_layer = lambda _model: (
            [(f"layers.{idx}", DummyBlock()) for idx in range(num_layers)] + [("mtp.0", DummyBlock())]
        )

        generator = adapter.generate_model_forward(model, torch.zeros(1, 4, dtype=torch.long))
        request = next(generator)
        for _ in range(num_layers):
            request = generator.send(torch.ones(1, 4, 2, DIM))
        assert request.name == "mtp.0"
        return captured["main_hidden"]

    def test_remapped_duplicates_each_contribute_a_hidden(self):
        adapter = make_adapter(
            SimpleNamespace(
                num_hidden_layers=3,
                n_mtp_layers=1,
                dim=DIM,
                dspark_target_layer_ids=(14, 29, 44),
                dspark_effective_target_layer_ids=(2, 2, 2),
            )
        )

        main_hidden = self._run_until_mtp(adapter, num_layers=3)

        assert main_hidden.shape[-1] == DIM * 3

    def test_untruncated_targets_keep_the_same_width(self):
        adapter = make_adapter(
            SimpleNamespace(
                num_hidden_layers=3,
                n_mtp_layers=1,
                dim=DIM,
                dspark_target_layer_ids=(0, 1, 2),
                dspark_effective_target_layer_ids=(0, 1, 2),
            )
        )

        main_hidden = self._run_until_mtp(adapter, num_layers=3)

        assert main_hidden.shape[-1] == DIM * 3
