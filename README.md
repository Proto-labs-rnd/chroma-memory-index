# Chroma Memory Index

Chroma-backed semantic memory & skill index for AI agents. Designed for **ARM64** (Raspberry Pi 5) with streaming upsert to prevent OOM.

## Features

- 🔍 Semantic search across memory files and skill/tool docs
- 📊 Dual collections: `proto-memory` + `proto-skills`
- 🔄 Incremental indexing (only new/changed docs)
- 🛡️ ARM64-safe: streaming chunk upsert + `gc.collect()` to prevent SIGKILL
- ⚡ Ollama embeddings (`nomic-embed-text`) — 100% local, no API keys
- 🔒 All configuration via environment variables (no hardcoded secrets)

## Install

```bash
pip install -e .
```

## Usage

```bash
# Daily-safe refresh (recommended)
chroma-memory-index index --memory --incremental
chroma-memory-index index --skills --incremental

# Full rebuild
chroma-memory-index index --all

# Search
chroma-memory-index query "routing experiment accuracy"
chroma-memory-index query "docker" --collection skills -n 10

# Stats
chroma-memory-index stats
```

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_HOST` | `127.0.0.1` | ChromaDB host |
| `CHROMA_PORT` | `8001` | ChromaDB port |
| `CHROMA_USER` | *(empty)* | ChromaDB username |
| `CHROMA_PASS` | *(empty)* | ChromaDB password |
| `CHROMA_SSL` | `false` | Enable SSL |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBED_BATCH_SIZE` | `4` | Batch size for embedding |
| `EMBED_TIMEOUT` | `30` | Timeout per batch (seconds) |
| `EMBED_RETRIES` | `1` | Retry count per batch |
| `WORKSPACE_PATH` | *(see config.py)* | Workspace root path |

## Architecture

```
src/chroma_memory_index/
  __init__.py    — Package metadata
  config.py      — Dataclass-based configuration (env-driven)
  collector.py   — File collection (memory, skills, experiments)
  embed.py       — Ollama embedding with ARM64-safe retry logic
  core.py        — Index/query/stats operations
  cli.py         — CLI entry point with subcommands
```

## Requirements

- Python 3.10+
- ChromaDB server running (or use with an existing instance)
- Ollama with `nomic-embed-text` model

## License

MIT
