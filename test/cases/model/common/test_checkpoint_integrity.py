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
"""

import json

import pytest

from msmodelslim.model.common.checkpoint_integrity import (
    ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT,
    CheckpointIntegrity,
    _format_int_ranges,
    check_checkpoint_or_raise,
    clamp_block_count,
    resolve_block_count,
    scan_checkpoint,
)
from msmodelslim.utils.exception import InvalidModelError

SHARED_KEYS = ("embed.weight", "norm.weight", "head.weight")


def _build_checkpoint(tmp_path, num_layers=6, num_mtp=2, present_files=None):
    """造一个 index.json 指向 N 个分片、但只在磁盘上放 present_files 的权重目录。"""
    weight_map = {}
    for key in SHARED_KEYS:
        weight_map[key] = "model-00000.safetensors"
    for idx in range(num_layers):
        weight_map[f"layers.{idx}.attn.wkv.weight"] = f"model-{idx + 1:05d}.safetensors"
        weight_map[f"layers.{idx}.ffn.w1.weight"] = f"model-{idx + 1:05d}.safetensors"
    for idx in range(num_mtp):
        weight_map[f"mtp.{idx}.attn.wkv.weight"] = f"model-{num_layers + idx + 1:05d}.safetensors"

    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")

    if present_files is None:
        present_files = sorted(set(weight_map.values()))
    for name in present_files:
        (tmp_path / name).write_bytes(b"")
    return tmp_path


class TestFormatIntRanges:
    def test_collapses_consecutive_values(self):
        assert _format_int_ranges([3, 4, 5, 9, 11, 12]) == "3-5,9,11-12"

    def test_empty(self):
        assert _format_int_ranges([]) == ""


class TestScanCheckpoint:
    def test_complete_checkpoint_reports_no_missing_file(self, tmp_path):
        model_path = _build_checkpoint(tmp_path)

        integrity = scan_checkpoint(model_path)

        assert integrity.is_complete
        assert integrity.missing_files == []

    def test_missing_index_is_treated_as_complete(self, tmp_path):
        integrity = scan_checkpoint(tmp_path)

        assert integrity.is_complete
        assert integrity.total_files == 0

    def test_truncated_checkpoint_lists_missing_files_and_usable_layers(self, tmp_path):
        # 保留 shared(00000)、layers.0-2(00001-00003) 与两个 mtp 分片，丢掉 layers.3-5
        model_path = _build_checkpoint(
            tmp_path,
            present_files=[
                "model-00000.safetensors",
                "model-00001.safetensors",
                "model-00002.safetensors",
                "model-00003.safetensors",
                "model-00007.safetensors",
                "model-00008.safetensors",
            ],
        )

        integrity = scan_checkpoint(model_path)

        assert not integrity.is_complete
        assert integrity.missing_files == [
            "model-00004.safetensors",
            "model-00005.safetensors",
            "model-00006.safetensors",
        ]
        assert integrity.shared_weights_complete
        assert integrity.available_prefix_length("layers") == 3
        assert integrity.available_prefix_length("mtp") == 2
        assert integrity.missing_blocks["layers"] == {3, 4, 5}

    def test_a_layer_is_unusable_when_only_one_of_its_shards_is_missing(self, tmp_path):
        weight_map = {
            "embed.weight": "shared.safetensors",
            "layers.0.attn.wkv.weight": "a.safetensors",
            "layers.0.ffn.w1.weight": "b.safetensors",
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (tmp_path / "shared.safetensors").write_bytes(b"")
        (tmp_path / "a.safetensors").write_bytes(b"")

        integrity = scan_checkpoint(tmp_path)

        assert integrity.available_prefix_length("layers") == 0
        assert integrity.missing_blocks["layers"] == {0}

    def test_missing_shared_weight_is_flagged(self, tmp_path):
        model_path = _build_checkpoint(tmp_path, present_files=["model-00001.safetensors"])

        integrity = scan_checkpoint(model_path)

        assert not integrity.shared_weights_complete

    def test_available_prefix_length_requires_a_gapless_prefix(self):
        integrity = CheckpointIntegrity(missing_files=["x"], present_blocks={"layers": {0, 1, 4}})

        assert integrity.available_prefix_length("layers") == 2


class TestCheckCheckpointOrRaise:
    def test_complete_checkpoint_passes(self, tmp_path):
        model_path = _build_checkpoint(tmp_path)

        assert check_checkpoint_or_raise(model_path).is_complete

    def test_raises_by_default_and_names_the_missing_files(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, raising=False)
        model_path = _build_checkpoint(tmp_path, present_files=["model-00000.safetensors"])

        with pytest.raises(InvalidModelError) as err:
            check_checkpoint_or_raise(model_path)

        assert "model-00001.safetensors" in str(err.value)
        assert ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT in str(err.value)

    def test_returns_integrity_when_truncation_is_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, "1")
        model_path = _build_checkpoint(
            tmp_path,
            present_files=[
                "model-00000.safetensors",
                "model-00001.safetensors",
                "model-00007.safetensors",
                "model-00008.safetensors",
            ],
        )

        integrity = check_checkpoint_or_raise(model_path)

        assert not integrity.is_complete
        assert integrity.available_prefix_length("layers") == 1

    def test_still_raises_when_shared_weights_are_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, "1")
        model_path = _build_checkpoint(tmp_path, present_files=["model-00001.safetensors"])

        with pytest.raises(InvalidModelError) as err:
            check_checkpoint_or_raise(model_path)

        assert "embedding" in str(err.value)


class TestClampBlockCount:
    def test_complete_checkpoint_keeps_the_declared_count(self):
        integrity = CheckpointIntegrity()

        assert resolve_block_count(integrity, "layers", 61) == 61
        assert clamp_block_count(integrity, "layers", 61, "num_hidden_layers") == 61

    def test_clamps_to_the_available_prefix(self):
        integrity = CheckpointIntegrity(missing_files=["x"], present_blocks={"layers": {0, 1, 2}})

        assert clamp_block_count(integrity, "layers", 61, "num_hidden_layers") == 3

    def test_never_grows_beyond_the_declared_count(self):
        integrity = CheckpointIntegrity(missing_files=["x"], present_blocks={"layers": {0, 1, 2, 3}})

        assert clamp_block_count(integrity, "layers", 2, "num_hidden_layers") == 2

    def test_raises_when_not_even_the_first_block_is_usable(self):
        integrity = CheckpointIntegrity(missing_files=["x"], present_blocks={"layers": {1, 2}})

        with pytest.raises(InvalidModelError):
            clamp_block_count(integrity, "layers", 61, "num_hidden_layers")
