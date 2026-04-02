"""Tests for chroma_memory_index.collector module."""

import os
from pathlib import Path
from unittest.mock import patch

from chroma_memory_index.collector import collect_memory_files, collect_skill_files, content_hash
from chroma_memory_index.config import IndexConfig


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different(self):
        assert content_hash("foo") != content_hash("bar")

    def test_length(self):
        assert len(content_hash("x")) == 12

    def test_empty_string(self):
        h = content_hash("")
        assert isinstance(h, str) and len(h) == 12


class TestCollectMemoryFiles:
    def test_empty_workspace(self, tmp_path):
        cfg = IndexConfig(workspace=tmp_path)
        docs = collect_memory_files(cfg)
        # Should work even with no files
        assert isinstance(docs, list)

    def test_with_memory_md(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("# My memory\nImportant stuff")
        cfg = IndexConfig(workspace=tmp_path)
        docs = collect_memory_files(cfg)
        paths = [p for p, _, _ in docs]
        assert "MEMORY.md" in paths

    def test_with_daily_memory(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "2026-04-01.md").write_text("Today I learned X")
        (mem_dir / "2026-04-02.md").write_text("Today I learned Y")
        cfg = IndexConfig(workspace=tmp_path)
        docs = collect_memory_files(cfg)
        paths = [p for p, _, _ in docs]
        assert any("2026-04-01" in p for p in paths)
        assert any("2026-04-02" in p for p in paths)

    def test_with_experiment_results(self, tmp_path):
        exp_dir = tmp_path / "experiments" / "2026-04-01-test-exp"
        exp_dir.mkdir(parents=True)
        (exp_dir / "RESULT.md").write_text("Experiment succeeded")
        cfg = IndexConfig(workspace=tmp_path)
        docs = collect_memory_files(cfg)
        paths = [p for p, _, _ in docs]
        assert any("test-exp" in p for p in paths)

    def test_respects_limits(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        for i in range(100):
            (mem_dir / f"2026-01-{i:02d}.md").write_text(f"Day {i}")
        cfg = IndexConfig(workspace=tmp_path, memory_file_limit=10)
        docs = collect_memory_files(cfg)
        daily = [p for p, _, _ in docs if p.startswith("memory/2026")]
        assert len(daily) <= 10


class TestCollectSkillFiles:
    def test_empty_workspace(self, tmp_path):
        cfg = IndexConfig(workspace=tmp_path)
        docs = collect_skill_files(cfg)
        assert isinstance(docs, list)

    def test_with_tools(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "my-script.py").write_text("# My awesome tool\nprint('hello')")
        cfg = IndexConfig(workspace=tmp_path)
        docs = collect_skill_files(cfg)
        paths = [p for p, _, _ in docs]
        assert any("my-script.py" in p for p in paths)

    def test_experiments_md(self, tmp_path):
        (tmp_path / "EXPERIMENTS.md").write_text("# Experiments backlog")
        cfg = IndexConfig(workspace=tmp_path)
        docs = collect_skill_files(cfg)
        paths = [p for p, _, _ in docs]
        assert "EXPERIMENTS.md" in paths

    def test_tools_md(self, tmp_path):
        (tmp_path / "TOOLS.md").write_text("# Tools catalog")
        cfg = IndexConfig(workspace=tmp_path)
        docs = collect_skill_files(cfg)
        paths = [p for p, _, _ in docs]
        assert "TOOLS.md" in paths
