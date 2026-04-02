"""Tests for chroma_memory_index.embed module."""

from unittest.mock import MagicMock, patch

import pytest

from chroma_memory_index.config import EmbedConfig
from chroma_memory_index.embed import embed_texts


class TestEmbedTexts:
    def test_empty_input(self):
        assert embed_texts([]) == []

    def test_single_text(self):
        config = EmbedConfig()
        with patch("chroma_memory_index.embed._call_ollama", return_value=[[0.1] * 10]):
            result = embed_texts(["hello"], config)
        assert len(result) == 1
        assert len(result[0]) == 10

    def test_multiple_texts(self):
        config = EmbedConfig(batch_size=2)
        with patch("chroma_memory_index.embed._call_ollama", return_value=[[0.1] * 10, [0.2] * 10]):
            result = embed_texts(["hello", "world"], config)
        assert len(result) == 2

    def test_truncation(self):
        config = EmbedConfig(max_text_length=10)
        with patch("chroma_memory_index.embed._call_ollama") as mock_call:
            mock_call.return_value = [[0.1] * 10]
            embed_texts(["a" * 100], config)
        # Check that text was truncated
        call_args = mock_call.call_args
        assert len(call_args[0][0][0]) == 10

    def test_retry_on_failure(self):
        config = EmbedConfig(retries=1, batch_size=1)
        with patch("chroma_memory_index.embed._call_ollama") as mock_call:
            mock_call.side_effect = [Exception("fail"), [[0.1] * 10]]
            result = embed_texts(["hello"], config)
        assert len(result) == 1

    def test_permanent_failure_raises(self):
        config = EmbedConfig(retries=0, batch_size=1)
        with patch("chroma_memory_index.embed._call_ollama", side_effect=Exception("fail")):
            # Falls back to single-item, which also fails
            with pytest.raises(RuntimeError, match="embedding failed permanently"):
                embed_texts(["hello"], config)
