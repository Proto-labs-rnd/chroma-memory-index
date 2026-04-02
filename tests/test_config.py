"""Tests for chroma_memory_index.config module."""

import os
from chroma_memory_index.config import ChromaConfig, EmbedConfig, IndexConfig


class TestChromaConfig:
    def test_defaults(self):
        cfg = ChromaConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8001
        assert cfg.password == ""

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("CHROMA_HOST", "10.0.0.1")
        monkeypatch.setenv("CHROMA_PORT", "9000")
        monkeypatch.setenv("CHROMA_USER", "admin")
        monkeypatch.setenv("CHROMA_PASS", "secret")
        cfg = ChromaConfig.from_env()
        assert cfg.host == "10.0.0.1"
        assert cfg.port == 9000
        assert cfg.username == "admin"
        assert cfg.password == "secret"

    def test_ssl_env(self, monkeypatch):
        monkeypatch.setenv("CHROMA_SSL", "true")
        assert ChromaConfig.from_env().ssl is True
        monkeypatch.setenv("CHROMA_SSL", "nope")
        assert ChromaConfig.from_env().ssl is False


class TestEmbedConfig:
    def test_defaults(self):
        cfg = EmbedConfig()
        assert cfg.model == "nomic-embed-text"
        assert cfg.batch_size == 4

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("EMBED_MODEL", "custom-model")
        monkeypatch.setenv("EMBED_BATCH_SIZE", "8")
        cfg = EmbedConfig.from_env()
        assert cfg.model == "custom-model"
        assert cfg.batch_size == 8


class TestIndexConfig:
    def test_defaults(self):
        cfg = IndexConfig()
        assert cfg.memory_collection == "proto-memory"
        assert cfg.skills_collection == "proto-skills"
        assert cfg.query_results == 5

    def test_from_env(self, monkeypatch):
        cfg = IndexConfig.from_env()
        assert isinstance(cfg, IndexConfig)

    def test_workspace_env(self, monkeypatch):
        monkeypatch.setenv("WORKSPACE_PATH", "/tmp/test-workspace")
        cfg = IndexConfig()
        assert str(cfg.workspace) == "/tmp/test-workspace"
