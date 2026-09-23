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

from msmodelslim.app.naive_quantization.save_path_guard import (
    ENV_VAR_KEEP_FAILED_OUTPUT,
    INCOMPLETE_MARKER_NAME,
    SavePathGuard,
)


@pytest.fixture(autouse=True)
def _no_keep_env(monkeypatch):
    monkeypatch.delenv(ENV_VAR_KEEP_FAILED_OUTPUT, raising=False)


def _names(path):
    return sorted(entry.name for entry in path.iterdir())


class TestSavePathGuardSuccess:
    def test_marker_exists_during_the_run_and_is_gone_afterwards(self, tmp_path):
        with SavePathGuard(tmp_path) as guard:
            assert guard.marker_path.is_file()
            payload = json.loads(guard.marker_path.read_text(encoding="utf-8"))
            assert payload["save_path"] == str(tmp_path)
            (tmp_path / "quant_model_weights.safetensors").write_bytes(b"w")

        assert _names(tmp_path) == ["quant_model_weights.safetensors"]

    def test_creates_the_directory_when_it_does_not_exist(self, tmp_path):
        target = tmp_path / "out"

        with SavePathGuard(target):
            assert target.is_dir()

        assert target.is_dir()


class TestSavePathGuardFailure:
    def test_removes_only_what_the_run_created(self, tmp_path):
        (tmp_path / "pre_existing.json").write_text("{}", encoding="utf-8")

        with pytest.raises(RuntimeError):
            with SavePathGuard(tmp_path):
                (tmp_path / "quant_model_weights-00001.safetensors").write_bytes(b"w")
                (tmp_path / "quant_model_description.json").write_text("{}", encoding="utf-8")
                (tmp_path / "rank_0").mkdir()
                (tmp_path / "rank_0" / "part.safetensors").write_bytes(b"w")
                raise RuntimeError("boom")

        assert _names(tmp_path) == ["pre_existing.json"]

    def test_keeps_everything_when_the_escape_hatch_is_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR_KEEP_FAILED_OUTPUT, "1")

        with pytest.raises(RuntimeError):
            with SavePathGuard(tmp_path):
                (tmp_path / "quant_model_description.json").write_text("{}", encoding="utf-8")
                raise RuntimeError("boom")

        assert _names(tmp_path) == [INCOMPLETE_MARKER_NAME, "quant_model_description.json"]

    def test_does_not_swallow_the_exception(self, tmp_path):
        with pytest.raises(ValueError, match="original"):
            with SavePathGuard(tmp_path):
                raise ValueError("original")

    def test_a_leftover_marker_from_a_killed_run_is_reported_and_reset(self, tmp_path):
        (tmp_path / INCOMPLETE_MARKER_NAME).write_text("{}", encoding="utf-8")
        (tmp_path / "half_written.safetensors").write_bytes(b"w")

        with SavePathGuard(tmp_path):
            pass

        # 上一次被强杀留下的文件不属于本次运行，不会被删；标记在本次成功后清掉。
        assert _names(tmp_path) == ["half_written.safetensors"]
