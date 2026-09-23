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

    @property
    def is_complete(self) -> bool:
        return not self.missing_files

    def available_prefix_length(self, prefix: str) -> int:
        """`prefix` 下从 0 开始连续可用的 block 数量。"""
        present = self.present_blocks.get(prefix, set())
        length = 0
        while length in present:
            length += 1
        return length

    def describe(self, max_items: int = 8) -> str:
        """给用户看的一段摘要。"""
        shown = self.missing_files[:max_items]
        more = len(self.missing_files) - len(shown)
        files = ', '.join(shown) + (f' (+{more} more)' if more > 0 else '')
        parts = [f'{len(self.missing_files)}/{self.total_files} weight files are missing: {files}']
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
    if integrity.is_complete:
        return integrity

    present: Dict[str, Set[int]] = {}
    missing: Dict[str, Set[int]] = {}
    for weight_key, file_name in weight_map.items():
        block = _split_block_key(weight_key)
        if block is None:
            if not existing[file_name]:
                integrity.shared_weights_complete = False
            continue
        prefix, idx = block
        bucket = present if existing[file_name] else missing
        bucket.setdefault(prefix, set()).add(idx)
    # 一个 block 只要缺一个权重就不可用
    for prefix, indices in missing.items():
        present.setdefault(prefix, set()).difference_update(indices)
    integrity.present_blocks = present
    integrity.missing_blocks = missing
    return integrity


def resolve_block_count(
    integrity: CheckpointIntegrity,
    prefix: str,
    declared: int,
) -> int:
    """`prefix` 下可用的 block 数，上限为配置里声明的数量。"""
    if integrity.is_complete:
        return declared
    return min(declared, integrity.available_prefix_length(prefix))


def check_checkpoint_or_raise(model_path: Union[str, Path]) -> CheckpointIntegrity:
    """扫描权重目录；不完整且未开启截断模式时直接报错。

    Returns:
        CheckpointIntegrity: 扫描结果（完整时 `is_complete` 为 True）。
    """
    integrity = scan_checkpoint(model_path)
    if integrity.is_complete:
        return integrity

    summary = integrity.describe()
    if not allow_truncated_checkpoint():
        raise InvalidModelError(
            f'Incomplete checkpoint at {model_path}: {summary}',
            action=(
                'Please download the missing safetensors files. If the checkpoint was truncated on purpose '
                f'(a debug model with fewer layers), set {ENV_VAR_ALLOW_TRUNCATED_CHECKPOINT}=1 to quantize '
                'only the layers that are present -- the result is NOT numerically usable.'
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
        '%s is set: quantizing only the layers present on disk. '
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
    get_logger().warning('Truncating %s from %s to %s to match the checkpoint on disk', label, declared, available)
    return available
