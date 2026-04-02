"""Tests for chroma_memory_index.core module."""

from unittest.mock import MagicMock, patch

import pytest

from chroma_memory_index.config import EmbedConfig, IndexConfig
from chroma_memory_index.core import get_client, get_stats, index_collection, query_collection


class TestGetClient:
    def test_default_config(self):
        client = get_client()
        assert client is not None

    def test_custom_config(self):
        cfg = IndexConfig()
        client = get_client(cfg)
        assert client is not None


class TestGetStats:
    def test_found_collection(self):
        client = MagicMock()
        mock_col = MagicMock()
        mock_col.count.return_value = 42
        client.get_collection.return_value = mock_col
        stats = get_stats(client, ["test-col"])
        assert stats["test-col"] == 42

    def test_missing_collection(self):
        client = MagicMock()
        client.get_collection.side_effect = Exception("not found")
        stats = get_stats(client, ["missing"])
        assert stats["missing"] == -1


class TestIndexCollection:
    def test_empty_docs(self):
        client = MagicMock()
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        client.get_or_create_collection.return_value = mock_col

        with patch("chroma_memory_index.core.embed_texts", return_value=[[0.1] * 10]):
            n = index_collection(client, "test-col", [], incremental=False)
        assert n == 0

    def test_incremental_skips_existing(self):
        client = MagicMock()
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.get.return_value = {"ids": ["doc-abc123456789"]}
        client.get_or_create_collection.return_value = mock_col

        docs = [("path", "content that hashes to doc-abc123456789", "doc-abc123456789")]
        n = index_collection(client, "test-col", docs, incremental=True)
        assert n == 1  # all up-to-date
        mock_col.upsert.assert_not_called()

    def test_full_rebuild_indexes_all(self):
        client = MagicMock()
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        client.get_or_create_collection.return_value = mock_col

        docs = [("path.txt", "hello world", "doc-12345")]
        with patch("chroma_memory_index.core.embed_texts", return_value=[[0.1] * 10]):
            n = index_collection(client, "test-col", docs, incremental=False)
        assert n == 1
        mock_col.upsert.assert_called_once()


class TestQueryCollection:
    def test_missing_collection(self):
        client = MagicMock()
        client.get_collection.side_effect = Exception("not found")
        with pytest.raises(ValueError, match="not found"):
            query_collection(client, "missing", "test query")

    def test_successful_query(self):
        client = MagicMock()
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["doc1 text"]],
            "metadatas": [[{"source": "test.md"}]],
            "distances": [[0.2]],
        }
        client.get_collection.return_value = mock_col

        with patch("chroma_memory_index.core.embed_texts", return_value=[[0.1] * 10]):
            results = query_collection(client, "test-col", "test query", n=5)
        assert "documents" in results
        assert results["documents"][0][0] == "doc1 text"
