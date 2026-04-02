"""Tests for chroma_memory_index.cli module."""

from unittest.mock import patch, MagicMock

from chroma_memory_index.cli import main, build_parser


class TestBuildParser:
    def test_version_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_index_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["index", "--memory", "--incremental"])
        assert args.command == "index"
        assert args.memory is True
        assert args.incremental is True

    def test_query_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["query", "test search", "--collection", "skills", "-n", "3"])
        assert args.command == "query"
        assert args.text == "test search"
        assert args.collection == "skills"
        assert args.top == 3

    def test_stats_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["stats"])
        assert args.command == "stats"

    def test_no_command_returns_1(self):
        # No subcommand → prints help, returns 1
        result = main([])
        assert result == 1


class TestMainIndex:
    def test_index_all_default(self, tmp_path):
        with patch("chroma_memory_index.cli.get_client") as mock_gc, \
             patch("chroma_memory_index.cli.collect_memory_files", return_value=[]), \
             patch("chroma_memory_index.cli.collect_skill_files", return_value=[]), \
             patch("chroma_memory_index.cli.index_collection", return_value=0):
            result = main(["--workspace", str(tmp_path), "index"])
            assert result == 0


import pytest
