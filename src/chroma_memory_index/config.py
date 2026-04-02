"""Configuration for chroma-memory-index."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ChromaConfig:
    """ChromaDB connection settings."""

    host: str = "127.0.0.1"
    port: int = 8001
    username: str = ""
    password: str = ""
    ssl: bool = False

    @classmethod
    def from_env(cls) -> "ChromaConfig":
        return cls(
            host=os.getenv("CHROMA_HOST", "127.0.0.1"),
            port=int(os.getenv("CHROMA_PORT", "8001")),
            username=os.getenv("CHROMA_USER", ""),
            password=os.getenv("CHROMA_PASS", ""),
            ssl=os.getenv("CHROMA_SSL", "").lower() in ("1", "true", "yes"),
        )


@dataclass(frozen=True)
class EmbedConfig:
    """Embedding model settings."""

    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "nomic-embed-text"
    batch_size: int = 4
    timeout: int = 30
    retries: int = 1
    arm64_safe_limit: int = 4
    max_text_length: int = 2000

    @classmethod
    def from_env(cls) -> "EmbedConfig":
        return cls(
            ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            batch_size=int(os.getenv("EMBED_BATCH_SIZE", "4")),
            timeout=int(os.getenv("EMBED_TIMEOUT", "30")),
            retries=int(os.getenv("EMBED_RETRIES", "1")),
        )


@dataclass(frozen=True)
class IndexConfig:
    """Top-level configuration."""

    workspace: Path = field(
        default_factory=lambda: Path(
            os.getenv("WORKSPACE_PATH", "/mnt/shared-storage/openclaw/workspace-labs")
        )
    )
    memory_collection: str = "proto-memory"
    skills_collection: str = "proto-skills"
    memory_file_limit: int = 60
    tech_watch_limit: int = 15
    result_limit: int = 30
    memory_max_chars: int = 8000
    daily_max_chars: int = 6000
    tech_watch_max_chars: int = 4000
    result_max_chars: int = 4000
    skill_max_chars: int = 3000
    query_results: int = 5
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)

    @classmethod
    def from_env(cls) -> "IndexConfig":
        return cls(
            chroma=ChromaConfig.from_env(),
            embed=EmbedConfig.from_env(),
        )
