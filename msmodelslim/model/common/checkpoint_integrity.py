#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

不完整（截断）权重目录的探测与处理。

权重是逐层懒加载的，因此缺少某个 safetensors 分片时，报错会发生在校准跑到那一层
之后，且错误信息只是「文件不存在」。本模块在任何权重被读取之前，用
model.safetensors.index.json 与磁盘上实际存在的分片做一次比对：

  - 默认：立即抛出 InvalidModelError，列出缺失的分片与受影响的层区间；
  - 设置 MSMODELSLIM_ALLOW_TRUNCATED_CHECKPOINT=1：把 num_hidden_layers /
    n_mtp_layers 截断到「从 0 开始连续存在」的层数，只量化这些层。

截断模式只用于调试用的小权重（例如只保留前若干层的 DeepSeek-V4），产出的量化
权重在数值上没有意义，因此必须显式开启并伴随告警。
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import json_safe_load

ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT = 'MSMODELSLIM_ALLOW_TRUNCATED_CHECKPOINT'

#: 把一个权重名解析成 (block 前缀, block 序号)，例如 layers.17.attn.wkv.weight -> ("layers", 17)
_BLOCK_KEY_PATTERN = re.compile(r'^(?P<prefix>[A-Za-z_][\w.]*?)\.(?P<idx>\d+)\.')

_TRUE_VALUES = ('1', 'true', 'yes', 'on')


def allow_truncated_checkpoint() -> bool:
    """截断权重模式是否被显式开启。"""
    return os.getenv(ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT, '').strip().lower() in _TRUE_VALUES


@dataclass
class CheckpointIntegrity:
    """一次权重目录完整性扫描的结果。"""

    #: index 中引用、但磁盘上不存在的分片文件名（有序）
    missing_files: List[str] = field(default_factory=list)
    #: index 引用的分片总数
    total_files: int = 0
    #: block 前缀 -> 该前缀下完整存在的 block 序号集合
    present_blocks: Dict[str, Set[int]] = field(default_factory=dict)
    #: block 前缀 -> 该前缀下至少缺一个权重的 block 序号集合
    missing_blocks: Dict[str, Set[int]] = field(default_factory=dict)
    #: 不属于任何 block 的权重（embed/head/norm 等）是否完整
    shared_weights_complete: bool = True
    #: index 自身就没列出的 block 权重，"layers.4" -> ["hc_attn_fn", ...]
    missing_keys: Dict[str, List[str]] = field(default_factory=dict)
    #: index 里每个 block 实际列出的权重后缀，prefix -> {序号 -> 后缀集合}
    block_keys: Dict[str, Dict[int, Set[str]]] = field(default_factory=dict)
    #: index 覆盖不到配置声明的数量，prefix -> (可用数, 声明数)
    short_prefixes: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return not self.missing_files and not self.missing_keys and not self.short_prefixes

    def available_prefix_length(self, prefix: str) -> int:
        """`prefix` 下从 0 开始连续可用的 block 数量。"""
        present = self.present_blocks.get(prefix, set())
        length = 0
        while length in present:
            length += 1
        return length

    def describe(self, max_items: int = 8) -> str:
        """给用户看的一段摘要。"""
        parts: List[str] = []
        if self.missing_files:
            shown = self.missing_files[:max_items]
            more = len(self.missing_files) - len(shown)
            files = ', '.join(shown) + (f' (+{more} more)' if more > 0 else '')
            parts.append(f'{len(self.missing_files)}/{self.total_files} weight files are missing: {files}')
        if self.missing_keys:
            samples = []
            for name, suffixes in list(self.missing_keys.items())[:3]:
                listed = ', '.join(suffixes[:3]) + (f' (+{len(suffixes) - 3} more)' if len(suffixes) > 3 else '')
                samples.append(f'{name} has no {listed}')
            more = len(self.missing_keys) - len(samples)
            tail = f' (+{more} more block(s))' if more > 0 else ''
            parts.append(
                f'model.safetensors.index.json does not list every weight of '
                f'{len(self.missing_keys)} block(s): {"; ".join(samples)}{tail}'
            )
        for prefix, (available, declared) in sorted(self.short_prefixes.items()):
            parts.append(
                f'model.safetensors.index.json only covers {available} "{prefix}" block(s) '
                f'while the model declares {declared}'
            )
        for prefix in sorted(self.missing_blocks):
            missing = sorted(self.missing_blocks[prefix])
            if not missing:
                continue
            available = self.available_prefix_length(prefix)
            usable = f'blocks 0-{available - 1} are usable' if available > 0 else 'no block is usable'
            parts.append(f'"{prefix}" blocks {_format_int_ranges(missing)} are incomplete, {usable}')
        if not self.shared_weights_complete:
            parts.append('shared weights outside of any block (embedding / head / norm) are incomplete')
        return '. '.join(parts)


def _format_int_ranges(values: Sequence[int]) -> str:
    """[3,4,5,9] -> "3-5,9"。"""
    ranges: List[str] = []
    start = prev = None
    for value in values:
        if start is None:
            start = prev = value
            continue
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f'{start}-{prev}')
        start = prev = value
    if start is not None:
        ranges.append(str(start) if start == prev else f'{start}-{prev}')
    return ','.join(ranges)


def _split_block_key(weight_key: str) -> Optional[Tuple[str, int]]:
    match = _BLOCK_KEY_PATTERN.match(weight_key)
    if match is None:
        return None
    return match.group('prefix'), int(match.group('idx'))


def _load_weight_map(model_path: str) -> Dict[str, str]:
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    if not os.path.isfile(index_path):
        return {}
    return json_safe_load(index_path).get('weight_map', {})


def _split_block_suffix(weight_key: str) -> Optional[Tuple[str, int, str]]:
    parsed = _split_block_key(weight_key)
    if parsed is None:
        return None
    prefix, idx = parsed
    return prefix, idx, weight_key[len(f'{prefix}.{idx}.') :]


def _collect_block_keys(weight_map: Dict[str, str]) -> Dict[str, Dict[int, Set[str]]]:
    """index 里的权重名按 block 归档：prefix -> {序号 -> 该 block 的权重后缀集合}。"""
    blocks: Dict[str, Dict[int, Set[str]]] = {}
    for weight_key in weight_map:
        parsed = _split_block_suffix(weight_key)
        if parsed is None:
            continue
        prefix, idx, suffix = parsed
        blocks.setdefault(prefix, {}).setdefault(idx, set()).add(suffix)
    return blocks


def record_missing_block_keys(
    integrity: CheckpointIntegrity,
    prefix: str,
    expected_by_index: Dict[int, Set[str]],
) -> None:
    """用「这个 block 真正需要哪些权重」来复核 index，并把缺权重的 block 标为不可用。

    `expected_by_index` 由模型适配器给出（通常是在 meta device 上搭一个 block 后取
    named_parameters），因此这里不做任何关于命名规律的猜测 —— DeepSeek-V4 各层结构本
    来就不一致（compress_ratio 交替决定 attn.indexer 是否存在，前 n_hash_layers 层用
    ffn.gate.tid2eid、其余层用 ffn.gate.bias），按 key 出现规律去推「该有什么」必然误判。
    """
    listed = integrity.block_keys.get(prefix, {})
    for idx, expected in expected_by_index.items():
        present = listed.get(idx)
        if present is None:
            continue  # index 里整个 block 都没有，由 available_prefix_length 处理
        missing = expected - present
        if not missing:
            continue
        integrity.missing_keys[f'{prefix}.{idx}'] = sorted(missing)
        integrity.missing_blocks.setdefault(prefix, set()).add(idx)
        integrity.present_blocks.setdefault(prefix, set()).discard(idx)
    integrity.missing_keys = dict(sorted(integrity.missing_keys.items()))


def scan_checkpoint(model_path: Union[str, Path]) -> CheckpointIntegrity:
    """比对 model.safetensors.index.json 与磁盘上的分片，返回完整性扫描结果。"""
    model_path = str(model_path)
    weight_map = _load_weight_map(model_path)

    referenced_files = sorted(set(weight_map.values()))
    existing = {name: os.path.isfile(os.path.join(model_path, name)) for name in referenced_files}
    integrity = CheckpointIntegrity(
        missing_files=[name for name in referenced_files if not existing[name]],
        total_files=len(referenced_files),
    )

    blocks = _collect_block_keys(weight_map)
    integrity.block_keys = blocks

    present: Dict[str, Set[int]] = {prefix: set(per_idx) for prefix, per_idx in blocks.items()}
    missing: Dict[str, Set[int]] = {prefix: set() for prefix in blocks}
    for weight_key, file_name in weight_map.items():
        if existing[file_name]:
            continue
        block = _split_block_key(weight_key)
        if block is None:
            integrity.shared_weights_complete = False
            continue
        missing[block[0]].add(block[1])
    # 一个 block 只要缺一个权重就不可用
    for prefix, indices in missing.items():
        present[prefix].difference_update(indices)
    integrity.present_blocks = present
    integrity.missing_blocks = {prefix: indices for prefix, indices in missing.items() if indices}
    return integrity


def record_declared_block_count(integrity: CheckpointIntegrity, prefix: str, declared: int) -> None:
    """拿配置声明的 block 数去对 index 实际覆盖到的数量。

    截断权重未必表现为「文件缺失」：如果做截断的脚本顺手重写了
    model.safetensors.index.json，只把留下来的张量写进去，那么 index 引用的文件一个不少，
    只是根本没有 layers.4 及以后的条目。这时必须拿 config.json 声明的层数来比，否则这份
    权重会被当成完整的放过去，跑到第 4 层才报「layers.4.xxx 不在 index 里」。
    """
    if declared <= 0 or prefix not in integrity.block_keys:
        # index 里压根没有这个前缀（命名方式不同），无从判断，交给加载时报错。
        return
    available = integrity.available_prefix_length(prefix)
    if available < declared:
        integrity.short_prefixes[prefix] = (available, declared)


def resolve_block_count(
    integrity: CheckpointIntegrity,
    prefix: str,
    declared: int,
) -> int:
    """`prefix` 下可用的 block 数，上限为配置里声明的数量。"""
    if prefix not in integrity.block_keys:
        return declared
    return min(declared, integrity.available_prefix_length(prefix))


def check_checkpoint_or_raise(model_path: Union[str, Path]) -> CheckpointIntegrity:
    """扫描权重目录；不完整且未开启截断模式时直接报错。

    Returns:
        CheckpointIntegrity: 扫描结果（完整时 `is_complete` 为 True）。
    """
    return enforce_checkpoint_integrity(scan_checkpoint(model_path), model_path)


def enforce_checkpoint_integrity(
    integrity: CheckpointIntegrity,
    model_path: Union[str, Path],
) -> CheckpointIntegrity:
    """对扫描结果下结论：完整则放行，不完整则按截断模式报错或告警。

    与 `check_checkpoint_or_raise` 分开，是为了让调用方能在下结论之前先补充
    `record_missing_block_keys` 那一步的信息。
    """
    if integrity.is_complete:
        return integrity

    summary = integrity.describe()
    if not allow_truncated_checkpoint():
        raise InvalidModelError(
            f'Incomplete checkpoint at {model_path}: {summary}',
            action=(
                'Please download the missing safetensors files, or -- if weights are absent from '
                'model.safetensors.index.json itself, or the index stops short of the layer count in '
                'config.json -- re-check the script that produced this checkpoint. '
                f'If it was truncated on purpose (a debug model with fewer layers), set '
                f'{ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT}=1 to quantize only the layers that are complete -- '
                'the result is NOT numerically usable.'
            ),
        )
    if not integrity.shared_weights_complete:
        raise InvalidModelError(
            f'Incomplete checkpoint at {model_path}: {summary}',
            action=(
                'Weights outside of the decoder blocks (embedding / head / final norm) cannot be truncated. '
                'Please download the missing safetensors files.'
            ),
        )
    get_logger().warning('Incomplete checkpoint at %s: %s', model_path, summary)
    get_logger().warning(
        '%s is set: quantizing only the layers whose weights are complete. '
        'The produced weights are for pipeline debugging only, NOT for inference accuracy.',
        ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT,
    )
    return integrity


def clamp_block_count(
    integrity: CheckpointIntegrity,
    prefix: str,
    declared: int,
    label: str,
) -> int:
    """把声明的 block 数截断到磁盘上可用的数量，并在发生截断时告警。"""
    available = resolve_block_count(integrity, prefix, declared)
    if available == declared:
        return declared
    if available <= 0 < declared:
        raise InvalidModelError(
            f'No usable "{prefix}" block found in the checkpoint ({label} declares {declared}).',
            action='Please make sure at least the first decoder layer is fully present.',
        )
    get_logger().warning(
        'Truncating %s from %s to %s to match the weights the checkpoint actually provides',
        label,
        declared,
        available,
    )
    return available
