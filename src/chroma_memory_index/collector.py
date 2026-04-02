"""File collection utilities for memory and skill indexing."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .config import IndexConfig

logger = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    """Compute a short content hash for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def collect_memory_files(config: IndexConfig) -> list[tuple[str, str, str]]:
    """Collect (path, content, doc_id) for all memory files.

    Includes MEMORY.md, daily memory, tech-watch digests, and experiment results.
    """
    docs: list[tuple[str, str, str]] = []
    import glob

    # MEMORY.md
    mf = config.workspace / "MEMORY.md"
    if mf.exists():
        text = mf.read_text(errors="replace")[: config.memory_max_chars]
        docs.append(("MEMORY.md", text, "memory-main"))

    # Daily memory files
    daily_files = sorted(glob.glob(str(config.workspace / "memory" / "*.md")))
    for f in daily_files[-config.memory_file_limit :]:
        text = Path(f).read_text(errors="replace")[: config.daily_max_chars]
        name = Path(f).name
        docs.append((f"memory/{name}", text, f"daily-{content_hash(text)}"))

    # Tech watch digests
    tw_files = sorted(glob.glob(str(config.workspace / "memory" / "tech-watch" / "*.md")))
    for f in tw_files[-config.tech_watch_limit :]:
        text = Path(f).read_text(errors="replace")[: config.tech_watch_max_chars]
        name = Path(f).name
        docs.append((f"memory/tech-watch/{name}", text, f"watch-{content_hash(text)}"))

    # Experiment results
    result_files = sorted(glob.glob(str(config.workspace / "experiments" / "*" / "RESULT.md")))
    for f in result_files[-config.result_limit :]:
        text = Path(f).read_text(errors="replace")[: config.result_max_chars]
        exp_name = Path(f).parent.name
        docs.append((f"experiments/{exp_name}/RESULT.md", text, f"result-{exp_name}"))

    logger.info("collected %d memory documents", len(docs))
    return docs


def collect_skill_files(config: IndexConfig) -> list[tuple[str, str, str]]:
    """Collect skill/tool files for indexing.

    Includes tools/*.py, tools/*.sh, EXPERIMENTS.md, and TOOLS.md.
    """
    docs: list[tuple[str, str, str]] = []
    import glob

    for ext in ("*.py", "*.sh"):
        for f in sorted(glob.glob(str(config.workspace / "tools" / ext))):
            text = Path(f).read_text(errors="replace")[: config.skill_max_chars]
            name = Path(f).name
            desc_line = ""
            for line in text.split("\n")[:5]:
                line = line.strip()
                if line.startswith("#") and len(line) > 5:
                    desc_line = line.lstrip("#").strip()
                    break
            combined = f"{desc_line}\n\n{text}" if desc_line else text
            docs.append((f"tools/{name}", combined, f"skill-{content_hash(text)}"))

    # EXPERIMENTS.md
    ef = config.workspace / "EXPERIMENTS.md"
    if ef.exists():
        text = ef.read_text(errors="replace")[: config.daily_max_chars]
        docs.append(("EXPERIMENTS.md", text, "experiments-backlog"))

    # TOOLS.md
    tf = config.workspace / "TOOLS.md"
    if tf.exists():
        text = tf.read_text(errors="replace")[: config.daily_max_chars]
        docs.append(("TOOLS.md", text, "tools-catalog"))

    logger.info("collected %d skill documents", len(docs))
    return docs
