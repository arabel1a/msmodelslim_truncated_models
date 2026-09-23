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

save_path 的「要么完整、要么干净」保护。

量化过程会分多次往 save_path 写入（导出的量化配置、quant_model_description.json、
逐段落盘的 quant_model_weights-*.safetensors、copy 过来的 config.json 等）。中途失败时
这些文件会原样留在目录里，既看不出是半成品，也会污染下一次运行。

本模块提供一个上下文管理器：
  - 进入时记录目录里已有的条目，并写一个 .msmodelslim_incomplete 标记；
  - 正常结束时删除标记；
  - 异常结束时删除本次运行新产生的条目（绝不碰运行前就存在的文件），并保留标记文件
    以外的现场清理日志。设置 MSMODELSLIM_KEEP_FAILED_OUTPUT=1 可保留全部中间产物，
    此时标记文件也会留下，提醒目录不可用。
即使进程被强杀（OOM / kill），标记文件也会留在目录里说明这份产物不完整。
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

from msmodelslim.utils.logging import get_logger

ENV_VAR_KEEP_FAILED_OUTPUT = 'MSMODELSLIM_KEEP_FAILED_OUTPUT'
INCOMPLETE_MARKER_NAME = '.msmodelslim_incomplete'

_TRUE_VALUES = ('1', 'true', 'yes', 'on')


def keep_failed_output() -> bool:
    """失败后是否保留中间产物。"""
    return os.getenv(ENV_VAR_KEEP_FAILED_OUTPUT, '').strip().lower() in _TRUE_VALUES


class SavePathGuard:
    """保证 save_path 要么是一次完整的产物，要么回到运行前的样子。"""

    def __init__(self, save_path: Path, model_path: Optional[Path] = None):
        self.save_path = Path(save_path)
        self.model_path = model_path
        self.marker_path = self.save_path / INCOMPLETE_MARKER_NAME
        self._pre_existing: Set[str] = set()

    def __enter__(self) -> 'SavePathGuard':
        self._pre_existing = self._list_entries()
        self._pre_existing.discard(INCOMPLETE_MARKER_NAME)
        stale = self.marker_path.exists()
        self._write_marker(stale)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self._remove_marker()
            return False
        if keep_failed_output():
            get_logger().error(
                'Quantization failed. %s is set, so the incomplete output under %s is kept; '
                'it is NOT a usable model (see %s).',
                ENV_VAR_KEEP_FAILED_OUTPUT,
                self.save_path,
                INCOMPLETE_MARKER_NAME,
            )
            return False
        removed, leftover = self._remove_new_entries()
        if removed:
            get_logger().error(
                'Quantization failed. Removed %s incomplete output entrie(s) from %s: %s. '
                'Set %s=1 to keep them for debugging.',
                len(removed),
                self.save_path,
                ', '.join(removed),
                ENV_VAR_KEEP_FAILED_OUTPUT,
            )
        if leftover:
            # 清不干净就把标记留着，别让残留看起来像一份完整产物。
            get_logger().error(
                'Could not clean up %s under %s; the directory is left marked as incomplete by %s.',
                ', '.join(leftover),
                self.save_path,
                INCOMPLETE_MARKER_NAME,
            )
        else:
            self._remove_marker()
        return False

    def _list_entries(self) -> Set[str]:
        if not self.save_path.is_dir():
            return set()
        return set(os.listdir(self.save_path))

    def _write_marker(self, stale: bool) -> None:
        payload = {
            'pid': os.getpid(),
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': str(self.model_path) if self.model_path is not None else None,
            'save_path': str(self.save_path),
            'note': 'Quantization is in progress or was interrupted; this directory is not a usable model.',
        }
        try:
            self.save_path.mkdir(parents=True, exist_ok=True)
            with os.fdopen(
                os.open(self.marker_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w', encoding='utf-8'
            ) as marker:
                json.dump(payload, marker, indent=2)
        except OSError as err:
            get_logger().warning('Cannot write the in-progress marker %s: %s', self.marker_path, err)
            return
        if stale:
            get_logger().warning(
                'Found a leftover %s in %s: a previous run did not finish, its output was incomplete.',
                INCOMPLETE_MARKER_NAME,
                self.save_path,
            )

    def _remove_marker(self) -> None:
        try:
            self.marker_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as err:
            get_logger().warning('Cannot remove %s: %s', self.marker_path, err)

    def _remove_new_entries(self) -> Tuple[List[str], List[str]]:
        """删除本次运行新增的条目，返回 (已删除, 未能删除)。"""
        removed: List[str] = []
        leftover: List[str] = []
        for name in sorted(self._list_entries() - self._pre_existing):
            if name == INCOMPLETE_MARKER_NAME:
                continue
            entry = self.save_path / name
            try:
                if entry.is_symlink() or entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    continue
            except OSError as err:
                get_logger().warning('Cannot remove the incomplete output %s: %s', entry, err)
                leftover.append(name)
                continue
            removed.append(name)
        return removed, leftover
