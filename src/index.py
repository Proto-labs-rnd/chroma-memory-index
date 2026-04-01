#!/usr/bin/env python3
"""
Chroma-backed semantic memory & skill index for Proto.

Usage:
  python3 tools/chroma-memory-index.py --index-all       # Full reindex
  python3 tools/chroma-memory-index.py --index-memory     # Memory files only
  python3 tools/chroma-memory-index.py --index-skills     # Skills/tools only
  python3 tools/chroma-memory-index.py --query "routing experiment accuracy"
  python3 tools/chroma-memory-index.py --stats
"""

import argparse
import gc
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import chromadb
import httpx

# --- Config ---
CHROMA_HOST = "127.0.0.1"
CHROMA_PORT = 8001
CHROMA_USER = "admin"
CHROMA_PASS = "labs_chroma_prod_2026_secure"
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
WORKSPACE = Path("/mnt/shared-storage/openclaw/workspace-labs")
DEFAULT_EMBED_BATCH_SIZE = 4
DEFAULT_EMBED_TIMEOUT = 30
DEFAULT_RETRIES = 1
ARM64_SAFE_BATCH_LIMIT = 4

MEMORY_COLLECTION = "proto-memory"
SKILLS_COLLECTION = "proto-skills"


def get_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def embed(
    texts: list[str],
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    timeout: int = DEFAULT_EMBED_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    model: str = EMBED_MODEL,
) -> list[list[float]]:
    """Embed texts via Ollama /api/embed with ARM-friendly fallback.

    Strategy:
    - use smaller batches by default
    - retry failed/timeouting batches
    - if a batch still fails, fall back to embedding one document at a time
    """
    if not texts:
        return []

    all_embeds: list[list[float]] = []
    total = len(texts)
    effective_batch_size = max(1, min(batch_size, ARM64_SAFE_BATCH_LIMIT))

    for i in range(0, total, effective_batch_size):
        batch = [t[:2000] for t in texts[i:i + effective_batch_size]]
        batch_end = min(i + effective_batch_size, total)
        embedded = False
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                resp = httpx.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": model, "input": batch},
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
                resp.raise_for_status()
                embeddings = resp.json()["embeddings"]
                if len(embeddings) != len(batch):
                    raise RuntimeError(
                        f"embedding count mismatch: got {len(embeddings)} for batch of {len(batch)}"
                    )
                all_embeds.extend(embeddings)
                embedded = True
                break
            except Exception as e:
                last_error = e
                warn(
                    f"     ⚠️ batch {i+1}-{batch_end}/{total} failed"
                    f" (attempt {attempt+1}/{retries+1}): {e}"
                )
                time.sleep(min(2 * (attempt + 1), 5))

        if not embedded:
            warn(
                f"     ↪ falling back to single-item embedding for batch {i+1}-{batch_end}/{total}"
            )
            for j, text in enumerate(batch, start=i + 1):
                single_ok = False
                for attempt in range(retries + 1):
                    try:
                        resp = httpx.post(
                            f"{OLLAMA_URL}/api/embed",
                            json={"model": model, "input": [text]},
                            headers={"Content-Type": "application/json"},
                            timeout=timeout,
                        )
                        resp.raise_for_status()
                        embeddings = resp.json()["embeddings"]
                        if len(embeddings) != 1:
                            raise RuntimeError(
                                f"single embedding count mismatch: got {len(embeddings)}"
                            )
                        all_embeds.extend(embeddings)
                        single_ok = True
                        break
                    except Exception as e:
                        last_error = e
                        warn(
                            f"       ⚠️ doc {j}/{total} failed"
                            f" (attempt {attempt+1}/{retries+1}): {e}"
                        )
                        time.sleep(min(2 * (attempt + 1), 5))
                if not single_ok:
                    raise RuntimeError(
                        f"embedding failed permanently for doc {j}/{total}"
                    ) from last_error

        log(f"     ... {batch_end}/{total} embedded")

    return all_embeds


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


# --- Memory Indexing ---

def collect_memory_files() -> list[tuple[str, str, str]]:
    """Collect (path, content, doc_id) for all memory files."""
    docs = []
    
    # MEMORY.md
    mf = WORKSPACE / "MEMORY.md"
    if mf.exists():
        text = mf.read_text()[:8000]
        docs.append(("MEMORY.md", text, "memory-main"))
    
    # Daily memory files (last 30 days)
    for f in sorted(glob.glob(str(WORKSPACE / "memory" / "*.md")))[-60:]:
        text = Path(f).read_text()[:6000]
        name = Path(f).name
        docs.append((f"memory/{name}", text, f"daily-{content_hash(text)}"))
    
    # Tech watch digests
    for f in sorted(glob.glob(str(WORKSPACE / "memory" / "tech-watch" / "*.md")))[-15:]:
        text = Path(f).read_text()[:4000]
        name = Path(f).name
        docs.append((f"memory/tech-watch/{name}", text, f"watch-{content_hash(text)}"))
    
    # Experiment RESULT.md files (last 30)
    for f in sorted(glob.glob(str(WORKSPACE / "experiments" / "*" / "RESULT.md")))[-30:]:
        text = Path(f).read_text()[:4000]
        exp_name = Path(f).parent.name
        docs.append((f"experiments/{exp_name}/RESULT.md", text, f"result-{exp_name}"))
    
    return docs


def collect_skill_files() -> list[tuple[str, str, str]]:
    """Collect skill/tool files."""
    docs = []
    
    # tools/*.py and tools/*.sh
    for ext in ("*.py", "*.sh"):
        for f in sorted(glob.glob(str(WORKSPACE / "tools" / ext))):
            text = Path(f).read_text()[:3000]
            name = Path(f).name
            # Extract first comment/description line
            desc_line = ""
            for line in text.split("\n")[:5]:
                line = line.strip()
                if line.startswith('#') and len(line) > 5:
                    desc_line = line.lstrip('#').strip()
                    break
            combined = f"{desc_line}\n\n{text}" if desc_line else text
            docs.append((f"tools/{name}", combined, f"skill-{content_hash(text)}"))
    
    # EXPERIMENTS.md
    ef = WORKSPACE / "EXPERIMENTS.md"
    if ef.exists():
        text = ef.read_text()[:6000]
        docs.append(("EXPERIMENTS.md", text, "experiments-backlog"))
    
    # TOOLS.md
    tf = WORKSPACE / "TOOLS.md"
    if tf.exists():
        text = tf.read_text()[:6000]
        docs.append(("TOOLS.md", text, "tools-catalog"))
    
    return docs


def index_collection(
    client,
    collection_name: str,
    docs: list[tuple[str, str, str]],
    incremental: bool = False,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    timeout: int = DEFAULT_EMBED_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    model: str = EMBED_MODEL,
):
    """Index documents into a Chroma collection.
    
    If incremental=True, only embed docs whose content_hash is new or changed.
    """
    col = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    
    # Build lookup of existing docs
    existing_ids = set()
    if incremental and col.count() > 0:
        try:
            existing = col.get(include=["metadatas"])
            existing_ids = set(existing["ids"])
        except Exception:
            pass
    
    all_ids, all_texts, all_metas = [], [], []
    skipped = 0
    
    for path, text, doc_id in docs:
        if incremental and doc_id in existing_ids:
            skipped += 1
            continue
        all_ids.append(doc_id)
        all_texts.append(text[:2000])  # truncate for embedding
        all_metas.append({"source": path, "chars": len(text)})
    
    if not all_ids:
        log(f"   ⏭️  All {len(docs)} docs up-to-date (skipped)")
        return len(docs)
    
    if skipped:
        log(f"   ⏭️  Skipped {skipped} unchanged, embedding {len(all_ids)} new/changed")
    
    # Stream embed + upsert in chunks to avoid OOM on ARM64
    CHUNK = 4  # max docs per upsert call (ARM64 safe)
    all_embeds = []
    total_to_embed = len(all_ids)
    for chunk_start in range(0, total_to_embed, CHUNK):
        chunk_end = min(chunk_start + CHUNK, total_to_embed)
        chunk_ids = all_ids[chunk_start:chunk_end]
        chunk_texts = all_texts[chunk_start:chunk_end]
        chunk_metas = all_metas[chunk_start:chunk_end]
        
        chunk_embeds = embed(
            chunk_texts,
            batch_size=batch_size,
            timeout=timeout,
            retries=retries,
            model=model,
        )
        col.upsert(
            ids=chunk_ids,
            embeddings=chunk_embeds,
            documents=chunk_texts,
            metadatas=chunk_metas,
        )
        log(f"     ↪ upserted {chunk_start+1}-{chunk_end}/{total_to_embed}")
        del chunk_embeds, chunk_texts, chunk_ids, chunk_metas
        gc.collect()
    all_embeds = []  # not needed anymore, embeddings already upserted
    
    return len(all_ids)


def query_collection(client, collection_name: str, query: str, n: int = 5):
    """Query a collection and return results."""
    col = client.get_collection(name=collection_name)
    q_emb = embed([query], batch_size=1)[0]
    results = col.query(query_embeddings=[q_emb], n_results=n)
    return results


def main():
    parser = argparse.ArgumentParser(description="Chroma Memory & Skill Index")
    parser.add_argument("--index-all", action="store_true")
    parser.add_argument("--index-memory", action="store_true")
    parser.add_argument("--index-skills", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--collection", choices=["memory", "skills", "all"], default="all")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--incremental", action="store_true", help="Only embed new/changed docs")
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)
    parser.add_argument("--embed-timeout", type=int, default=DEFAULT_EMBED_TIMEOUT)
    parser.add_argument("--embed-retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--embed-model", type=str, default=EMBED_MODEL)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    
    client = get_client()
    
    if args.stats:
        for name in [MEMORY_COLLECTION, SKILLS_COLLECTION]:
            try:
                col = client.get_collection(name=name)
                print(f"  {name}: {col.count()} documents")
            except Exception:
                print(f"  {name}: not found")
        return
    
    if args.index_memory or args.index_all:
        log("📚 Indexing memory files...")
        docs = collect_memory_files()
        print(f"   Found {len(docs)} files")
        n = index_collection(
            client,
            MEMORY_COLLECTION,
            docs,
            incremental=args.incremental,
            batch_size=args.embed_batch_size,
            timeout=args.embed_timeout,
            retries=args.embed_retries,
            model=args.embed_model,
        )
        log(f"   ✅ Indexed {n} documents into '{MEMORY_COLLECTION}'")
    
    gc.collect()

    if args.index_skills or args.index_all:
        log("🔧 Indexing skill/tool files...")
        docs = collect_skill_files()
        print(f"   Found {len(docs)} files")
        n = index_collection(
            client,
            SKILLS_COLLECTION,
            docs,
            incremental=args.incremental,
            batch_size=args.embed_batch_size,
            timeout=args.embed_timeout,
            retries=args.embed_retries,
            model=args.embed_model,
        )
        log(f"   ✅ Indexed {n} documents into '{SKILLS_COLLECTION}'")
    
    if args.query:
        collections = []
        if args.collection in ("all", "memory"):
            collections.append(("memory", MEMORY_COLLECTION))
        if args.collection in ("all", "skills"):
            collections.append(("skills", SKILLS_COLLECTION))
        
        for label, col_name in collections:
            try:
                results = query_collection(client, col_name, args.query, args.n)
                print(f"\n🔍 {label} (top {args.n}):")
                for i, (doc, meta, dist) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )):
                    sim = 1 - dist
                    print(f"  {i+1}. [{sim:.2%}] {meta['source']}")
                    print(f"     {doc[:150].strip()}...")
            except Exception as e:
                print(f"\n🔍 {label}: error — {e}")


if __name__ == "__main__":
    main()
