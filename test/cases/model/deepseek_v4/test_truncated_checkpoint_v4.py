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

截断权重目录下 DeepSeekV4ModelAdapter 的层数收敛与产物 config.json 回写。
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from msmodelslim.model.common.checkpoint_integrity import ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT
from msmodelslim.model.deepseek_v4.model_adapter import DeepSeekV4ModelAdapter
from msmodelslim.utils.exception import InvalidModelError


def build_truncated_checkpoint(tmp_path, num_layers=6, kept_layers=3, num_mtp=2):
    """index.json 声明 num_layers 层，磁盘上只放前 kept_layers 层（外加 shared 与 mtp）。"""
    weight_map = {
        "embed.weight": "shared.safetensors",
        "norm.weight": "shared.safetensors",
        "head.weight": "shared.safetensors",
    }
    for idx in range(num_layers):
        weight_map[f"layers.{idx}.attn.wkv.weight"] = f"layer-{idx}.safetensors"
    for idx in range(num_mtp):
        weight_map[f"mtp.{idx}.attn.wkv.weight"] = f"mtp-{idx}.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    (tmp_path / "shared.safetensors").write_bytes(b"")
    for idx in range(kept_layers):
        (tmp_path / f"layer-{idx}.safetensors").write_bytes(b"")
    for idx in range(num_mtp):
        (tmp_path / f"mtp-{idx}.safetensors").write_bytes(b"")
    return tmp_path


def make_adapter(model_path):
    """绕开 __init__（会真的去读 config 与权重），只装配这几个测试需要的状态。"""
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    adapter.model_path = Path(model_path)
    return adapter


class TestApplyCheckpointTruncation:
    def test_raises_without_the_env_var(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, raising=False)
        adapter = make_adapter(build_truncated_checkpoint(tmp_path))
        args = SimpleNamespace(num_hidden_layers=6, n_mtp_layers=2)

        with pytest.raises(InvalidModelError):
            adapter.apply_checkpoint_truncation(args)

        assert args.num_hidden_layers == 6

    def test_clamps_layers_to_what_is_on_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, "1")
        adapter = make_adapter(build_truncated_checkpoint(tmp_path))
        args = SimpleNamespace(num_hidden_layers=6, n_mtp_layers=2)

        adapter.apply_checkpoint_truncation(args)

        assert args.num_hidden_layers == 3
        assert args.n_mtp_layers == 2
        assert args.checkpoint_truncated is True

    def test_clamps_mtp_when_its_shards_are_gone(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, "1")
        model_path = build_truncated_checkpoint(tmp_path, num_mtp=2)
        (model_path / "mtp-1.safetensors").unlink()
        adapter = make_adapter(model_path)
        args = SimpleNamespace(num_hidden_layers=6, n_mtp_layers=2)

        adapter.apply_checkpoint_truncation(args)

        assert args.n_mtp_layers == 1

    def test_complete_checkpoint_changes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, raising=False)
        adapter = make_adapter(build_truncated_checkpoint(tmp_path, num_layers=3, kept_layers=3))
        args = SimpleNamespace(num_hidden_layers=3, n_mtp_layers=2)

        adapter.apply_checkpoint_truncation(args)

        assert args.num_hidden_layers == 3
        assert not hasattr(args, "checkpoint_truncated")


class TestTruncatedConfigOverrides:
    def test_no_overrides_for_a_complete_checkpoint(self, tmp_path):
        adapter = make_adapter(tmp_path)
        adapter.config = SimpleNamespace(num_hidden_layers=6)

        assert adapter.get_truncated_config_overrides() == {}

    def test_overrides_carry_the_clamped_layer_count(self, tmp_path):
        adapter = make_adapter(tmp_path)
        adapter.config = SimpleNamespace(
            checkpoint_truncated=True,
            num_hidden_layers=3,
            n_mtp_layers=2,
            compress_ratios=[1, 1, 4, 128, 4, 128],
        )

        overrides = adapter.get_truncated_config_overrides()

        assert overrides["num_hidden_layers"] == 3
        assert overrides["n_mtp_layers"] == 2
        assert overrides["compress_ratios"] == [1, 1, 4]


class TestAscendV1SavePostprocess:
    def test_rewrites_the_exported_config_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, "1")
        (tmp_path / "model").mkdir()
        model_path = build_truncated_checkpoint(tmp_path / "model")
        save_path = tmp_path / "out"
        save_path.mkdir()
        config_file = save_path / "config.json"
        config_file.write_text(json.dumps({"num_hidden_layers": 6, "hidden_size": 8}), encoding="utf-8")

        adapter = make_adapter(model_path)
        args = SimpleNamespace(num_hidden_layers=6, n_mtp_layers=2, compress_ratios=[1] * 6)
        adapter.apply_checkpoint_truncation(args)
        adapter.config = args

        adapter.ascendv1_save_postprocess(model=None, save_directory=str(save_path))

        written = json.loads(config_file.read_text(encoding="utf-8"))
        assert written["num_hidden_layers"] == 3
        assert written["hidden_size"] == 8
        assert written["msmodelslim_truncated_from"]["missing_weight_files"]

    def test_does_nothing_for_a_complete_checkpoint(self, tmp_path):
        save_path = tmp_path / "out"
        save_path.mkdir()
        config_file = save_path / "config.json"
        config_file.write_text(json.dumps({"num_hidden_layers": 6}), encoding="utf-8")
        adapter = make_adapter(tmp_path)
        adapter.config = SimpleNamespace(num_hidden_layers=6)

        adapter.ascendv1_save_postprocess(model=None, save_directory=str(save_path))

        assert json.loads(config_file.read_text(encoding="utf-8")) == {"num_hidden_layers": 6}
