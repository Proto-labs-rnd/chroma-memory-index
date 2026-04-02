"""Core indexing and query logic for chroma-memory-index."""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

import chromadb

from .collector import collect_memory_files, collect_skill_files
from .config import IndexConfig
from .embed import embed_texts

logger = logging.getLogger(__name__)


def get_client(config: Optional[IndexConfig] = None) -> chromadb.HttpClient:
    """Create a ChromaDB client from configuration."""
    if config is None:
        config = IndexConfig()
    cc = config.chroma
    return chromadb.HttpClient(host=cc.host, port=cc.port)


def index_collection(
    client: chromadb.HttpClient,
    collection_name: str,
    docs: list[tuple[str, str, str]],
    config: Optional[IndexConfig] = None,
    incremental: bool = False,
) -> int:
    """Index documents into a Chroma collection.

    Args:
        client: ChromaDB client instance.
        collection_name: Target collection name.
        docs: List of (path, content, doc_id) tuples.
        config: Index configuration.
        incremental: If True, skip docs whose content hash hasn't changed.

    Returns:
        Number of documents indexed (new or updated).
    """
    if config is None:
        config = IndexConfig()

    col = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Check existing docs for incremental mode
    existing_ids: set[str] = set()
    if incremental and col.count() > 0:
        try:
            existing = col.get(include=["metadatas"])
            existing_ids = set(existing["ids"])
        except Exception:
            logger.warning("failed to fetch existing IDs, treating as full rebuild")

    all_ids, all_texts, all_metas = [], [], []
    skipped = 0

    for path, text, doc_id in docs:
        if incremental and doc_id in existing_ids:
            skipped += 1
            continue
        all_ids.append(doc_id)
        all_texts.append(text[: config.embed.max_text_length])
        all_metas.append({"source": path, "chars": len(text)})

    if not all_ids:
        logger.info("all %d docs up-to-date (skipped)", len(docs))
        return len(docs)

    if skipped:
        logger.info("skipped %d unchanged, embedding %d new/changed", skipped, len(all_ids))

    # Stream embed + upsert in chunks to avoid OOM on ARM64
    chunk_size = config.embed.arm64_safe_limit
    total = len(all_ids)

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_ids = all_ids[chunk_start:chunk_end]
        chunk_texts = all_texts[chunk_start:chunk_end]
        chunk_metas = all_metas[chunk_start:chunk_end]

        chunk_embeds = embed_texts(chunk_texts, config.embed)
        col.upsert(
            ids=chunk_ids,
            embeddings=chunk_embeds,
            documents=chunk_texts,
            metadatas=chunk_metas,
        )
        logger.debug("upserted %d-%d/%d", chunk_start + 1, chunk_end, total)
        del chunk_embeds, chunk_texts, chunk_ids, chunk_metas
        gc.collect()

    return len(all_ids)


def query_collection(
    client: chromadb.HttpClient,
    collection_name: str,
    query: str,
    config: Optional[IndexConfig] = None,
    n: int = 0,
) -> dict[str, Any]:
    """Query a collection and return results.

    Args:
        client: ChromaDB client instance.
        collection_name: Collection to query.
        query: Search query string.
        config: Index configuration.
        n: Number of results. Uses config default if 0.

    Returns:
        ChromaDB query results dict with 'documents', 'metadatas', 'distances'.

    Raises:
        ValueError: If collection doesn't exist.
    """
    if config is None:
        config = IndexConfig()
    if n <= 0:
        n = config.query_results

    try:
        col = client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(f"collection '{collection_name}' not found: {e}") from e

    q_emb = embed_texts([query], config.embed)[0]
    return col.query(query_embeddings=[q_emb], n_results=n)


def get_stats(client: chromadb.HttpClient, collection_names: list[str]) -> dict[str, int]:
    """Get document counts for specified collections.

    Returns:
        Dict mapping collection name to document count (-1 if not found).
    """
    stats: dict[str, int] = {}
    for name in collection_names:
        try:
            col = client.get_collection(name=name)
            stats[name] = col.count()
        except Exception:
            stats[name] = -1
    return stats
