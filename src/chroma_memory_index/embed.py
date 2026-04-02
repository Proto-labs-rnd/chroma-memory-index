"""Embedding utilities for chroma-memory-index."""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from .config import EmbedConfig

logger = logging.getLogger(__name__)


def embed_texts(
    texts: list[str],
    config: Optional[EmbedConfig] = None,
) -> list[list[float]]:
    """Embed texts via Ollama /api/embed with ARM-friendly fallback.

    Strategy:
    - use smaller batches by default
    - retry failed/timeout batches
    - if a batch still fails, fall back to embedding one document at a time

    Args:
        texts: List of text strings to embed.
        config: Embedding configuration. Uses defaults if not provided.

    Returns:
        List of embedding vectors.

    Raises:
        RuntimeError: If embedding fails permanently for any document.
    """
    if config is None:
        config = EmbedConfig()

    if not texts:
        return []

    all_embeds: list[list[float]] = []
    total = len(texts)
    effective_batch = max(1, min(config.batch_size, config.arm64_safe_limit))

    for i in range(0, total, effective_batch):
        batch = [t[: config.max_text_length] for t in texts[i : i + effective_batch]]
        batch_end = min(i + effective_batch, total)
        embedded = False
        last_error: Optional[Exception] = None

        for attempt in range(config.retries + 1):
            try:
                embeddings = _call_ollama(batch, config)
                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        f"embedding count mismatch: got {len(embeddings)} "
                        f"for batch of {len(batch)}"
                    )
                all_embeds.extend(embeddings)
                embedded = True
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "batch %d-%d/%d failed (attempt %d/%d): %s",
                    i + 1,
                    batch_end,
                    total,
                    attempt + 1,
                    config.retries + 1,
                    e,
                )
                time.sleep(min(2 * (attempt + 1), 5))

        if not embedded and last_error is not None:
            logger.info(
                "falling back to single-item embedding for batch %d-%d", i + 1, batch_end
            )
            for j, text in enumerate(batch, start=i + 1):
                single_ok = False
                for attempt in range(config.retries + 1):
                    try:
                        embeddings = _call_ollama([text], config)
                        if len(embeddings) != 1:
                            raise RuntimeError(
                                f"single embedding count mismatch: got {len(embeddings)}"
                            )
                        all_embeds.extend(embeddings)
                        single_ok = True
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(
                            "doc %d/%d failed (attempt %d/%d): %s",
                            j,
                            total,
                            attempt + 1,
                            config.retries + 1,
                            e,
                        )
                        time.sleep(min(2 * (attempt + 1), 5))
                if not single_ok:
                    raise RuntimeError(
                        f"embedding failed permanently for doc {j}/{total}"
                    ) from last_error

        logger.debug("embedded %d-%d/%d", i + 1, batch_end, total)

    return all_embeds


def _call_ollama(texts: list[str], config: EmbedConfig) -> list[list[float]]:
    """Make a single embedding call to Ollama."""
    resp = httpx.post(
        f"{config.ollama_url}/api/embed",
        json={"model": config.model, "input": texts},
        headers={"Content-Type": "application/json"},
        timeout=config.timeout,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]
