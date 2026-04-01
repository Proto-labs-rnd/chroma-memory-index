# Chroma Memory Index

Chroma-backed semantic memory & skill index for AI agents. Designed for **ARM64** (Raspberry Pi 5, etc.) with streaming upsert to prevent OOM.

## Features

- 🔍 Semantic search across memory files and skill/tool docs
- 📊 Dual collections: `proto-memory` + `proto-skills`
- 🔄 Incremental indexing (only new/changed docs)
- 🛡️ ARM64-safe: streaming chunk upsert + `gc.collect()` to prevent SIGKILL
- ⚡ Ollama embeddings (`nomic-embed-text`) — 100% local, no API keys

## Usage

```bash
# Full reindex (memory + skills)
python3 src/index.py --index-all

# Incremental (only new/changed)
python3 src/index.py --index-all --incremental

# Single collection
python3 src/index.py --index-memory
python3 src/index.py --index-skills

# Search
python3 src/index.py --query "routing experiment accuracy" --collection all

# Stats
python3 src/index.py --stats
```

## Architecture

```
┌─────────────┐    embed (Ollama)    ┌──────────────┐
│ memory/     │ ──────────────────►  │              │
│ tools/      │    batch_size=4      │  Chroma DB   │
│ EXPERIMENTS │    chunk_upsert=4    │  (HTTP API)  │
│ TOOLS.md    │    gc per chunk      │              │
└─────────────┘                      └──────────────┘
```

### ARM64 OOM Fix

The `--index-all` command processes documents in **streaming chunks**:
1. Embed batch of 4 texts via Ollama
2. Upsert to Chroma immediately
3. `del` + `gc.collect()` — free RAM
4. Next chunk

This prevents the SIGKILL that occurred when loading all embeddings in memory before upserting.

## Requirements

- Python 3.10+
- `chromadb` (HTTP client mode)
- `httpx`
- Ollama running with `nomic-embed-text` model
- Chroma DB server (e.g., `chromadb/chroma:latest` via Docker)

## Configuration

Default connection:
- Chroma: `127.0.0.1:8001` (user: `admin`)
- Ollama: `localhost:11434` (model: `nomic-embed-text`)

Override with `--chroma-host`, `--chroma-port`, `--embed-model` flags.

## License

MIT
