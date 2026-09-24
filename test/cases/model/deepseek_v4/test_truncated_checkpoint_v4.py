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
from unittest.mock import patch

import pytest
import torch
from torch import nn

from msmodelslim.model.common.checkpoint_integrity import ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT
from msmodelslim.model.deepseek_v4 import model as v4_model
from msmodelslim.model.deepseek_v4.model import Attention, Block, ModelArgs
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
        assert "are missing" in written["msmodelslim_truncated_from"]["reason"]

    def test_does_nothing_for_a_complete_checkpoint(self, tmp_path):
        save_path = tmp_path / "out"
        save_path.mkdir()
        config_file = save_path / "config.json"
        config_file.write_text(json.dumps({"num_hidden_layers": 6}), encoding="utf-8")
        adapter = make_adapter(tmp_path)
        adapter.config = SimpleNamespace(num_hidden_layers=6)

        adapter.ascendv1_save_postprocess(model=None, save_directory=str(save_path))

        assert json.loads(config_file.read_text(encoding="utf-8")) == {"num_hidden_layers": 6}


REAL_LAYERS = 6


def fresh_args(num_layers=REAL_LAYERS):
    """真实的 ModelArgs，只把专家数和序列长度缩小，结构差异原样保留。"""
    args = ModelArgs()
    args.num_hidden_layers = num_layers
    args.n_mtp_layers = 0
    args.n_routed_experts = 2
    args.max_seq_len = 512
    return args


def build_index_from_real_blocks(root: Path, num_layers=REAL_LAYERS, drop=()):
    """用真实的 Block 参数名造一份 index。

    这样各层之间天然带着结构差异（compress_ratio==4 才有 attn.indexer，前
    n_hash_layers 层用 ffn.gate.tid2eid、其余层用 ffn.gate.bias），正是不能被误判成
    「缺权重」的东西。
    """
    args = fresh_args(num_layers)
    weight_map = {"embed.weight": "shared.safetensors", "head.weight": "shared.safetensors"}
    with torch.device("meta"), patch.object(nn.Linear, "reset_parameters", lambda _self: None):
        for idx in range(num_layers):
            for name, _ in Block(idx, args).named_parameters():
                weight_map[f"layers.{idx}.{name}"] = "shared.safetensors"
    for key in drop:
        weight_map.pop(key)

    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "shared.safetensors").write_bytes(b"")
    return root


class TestExpectedLayerKeys:
    """index 少列了某一层的权重时（分片都在，键没了）的处理。"""

    def test_a_complete_index_is_not_flagged_despite_structural_differences(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, raising=False)
        adapter = make_adapter(build_index_from_real_blocks(tmp_path / "m"))
        args = fresh_args()

        adapter.apply_checkpoint_truncation(args)

        assert args.num_hidden_layers == REAL_LAYERS
        assert adapter.checkpoint_integrity.missing_keys == {}

    def test_raises_and_names_the_missing_weight(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, raising=False)
        adapter = make_adapter(build_index_from_real_blocks(tmp_path / "m", drop=("layers.4.hc_attn_fn",)))

        with pytest.raises(InvalidModelError) as err:
            adapter.apply_checkpoint_truncation(fresh_args())

        assert "layers.4 has no hc_attn_fn" in str(err.value)

    def test_truncates_to_the_layers_whose_weights_are_all_listed(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, "1")
        adapter = make_adapter(build_index_from_real_blocks(tmp_path / "m", drop=("layers.4.hc_attn_fn",)))
        args = fresh_args()

        adapter.apply_checkpoint_truncation(args)

        assert args.num_hidden_layers == 4
        assert adapter.checkpoint_integrity.missing_keys == {"layers.4": ["hc_attn_fn"]}

    def test_an_expert_weight_missing_from_one_layer_is_caught(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, "1")
        adapter = make_adapter(build_index_from_real_blocks(tmp_path / "m", drop=("layers.2.ffn.experts.0.w1.weight",)))
        args = fresh_args()

        adapter.apply_checkpoint_truncation(args)

        assert args.num_hidden_layers == 2


class TestMetaBuildScopeLeavesNoMetaTensorsBehind:
    """meta device 上取参数名，不能污染后面真正建出来的模型。

    model.py 的 precompute_freqs_cis 是 @lru_cache 的，按当前默认 device 建张量。
    曾经因为在 meta 上下文里第一次命中它，真实模型的 Attention.freqs_cis 也成了 meta
    张量，LoadProcessor 把 layers.0 搬到 npu 时抛
    NotImplementedError: Cannot copy out of meta tensor。
    """

    def test_a_module_built_after_the_scope_has_real_buffers(self):
        args = fresh_args()
        v4_model.precompute_freqs_cis.cache_clear()

        with DeepSeekV4ModelAdapter._meta_build_scope():
            meta_block = Attention(0, args)
        assert meta_block.freqs_cis.is_meta  # 作用域内确实是 meta

        after = Attention(0, args)

        assert not after.freqs_cis.is_meta
        assert not after.kv_cache.is_meta

    def test_caches_are_cleared_even_when_the_body_raises(self):
        args = fresh_args()
        v4_model.precompute_freqs_cis.cache_clear()

        with pytest.raises(RuntimeError):
            with DeepSeekV4ModelAdapter._meta_build_scope():
                Attention(0, args)
                raise RuntimeError("boom")

        assert not Attention(0, args).freqs_cis.is_meta

    def test_the_real_checkpoint_scan_leaves_no_meta_tensors(self, tmp_path, monkeypatch):
        """走完整的 apply_checkpoint_truncation，再建一个模块检查。"""
        monkeypatch.delenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, raising=False)
        v4_model.precompute_freqs_cis.cache_clear()
        adapter = make_adapter(build_index_from_real_blocks(tmp_path / "m"))

        adapter.apply_checkpoint_truncation(fresh_args())

        # 必须用与扫描时相同的 args —— rope 参数不同就是另一个缓存键，测不到东西。
        assert not Attention(0, fresh_args()).freqs_cis.is_meta


class TestIndexStoppingShortOfTheDeclaredLayerCount:
    """截断脚本顺手重写了 index：文件一个不缺，只是 index 到 layers.3 就没了。

    这种权重一度被当成完整的放过去，跑到第 4 层才报「layers.4.xxx 不在 index 里」。
    """

    @staticmethod
    def _adapter_and_args(tmp_path, covered=4, declared=REAL_LAYERS):
        adapter = make_adapter(build_index_from_real_blocks(tmp_path / "m", num_layers=covered))
        return adapter, fresh_args(declared)

    def test_nothing_is_missing_at_the_file_level(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, "1")
        adapter, args = self._adapter_and_args(tmp_path)

        adapter.apply_checkpoint_truncation(args)

        assert adapter.checkpoint_integrity.missing_files == []
        assert adapter.checkpoint_integrity.short_prefixes == {"layers": (4, REAL_LAYERS)}

    def test_raises_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, raising=False)
        adapter, args = self._adapter_and_args(tmp_path)

        with pytest.raises(InvalidModelError) as err:
            adapter.apply_checkpoint_truncation(args)

        assert f'only covers 4 "layers" block(s) while the model declares {REAL_LAYERS}' in str(err.value)

    def test_clamps_to_what_the_index_covers(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, "1")
        adapter, args = self._adapter_and_args(tmp_path)

        adapter.apply_checkpoint_truncation(args)

        assert args.num_hidden_layers == 4
        assert args.checkpoint_truncated is True

    def test_an_index_covering_every_declared_layer_is_untouched(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, raising=False)
        adapter, args = self._adapter_and_args(tmp_path, covered=REAL_LAYERS)

        adapter.apply_checkpoint_truncation(args)

        assert args.num_hidden_layers == REAL_LAYERS
        assert adapter.checkpoint_integrity.short_prefixes == {}
