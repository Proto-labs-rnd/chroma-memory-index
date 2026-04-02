# Changelog

## v1.0.0 (2026-04-02)

### Added
- Proper package layout (`src/chroma_memory_index/`)
- CLI with subcommands: `index`, `query`, `stats`
- Environment-variable configuration (no hardcoded secrets)
- ARM64-safe embedding with retry and single-item fallback
- Comprehensive test suite (30+ tests)
- Type hints throughout
- Structured logging via `logging` module
- `pyproject.toml` with entry point

### Changed
- Migrated from flat `src/index.py` to modular package
- Replaced hardcoded ChromaDB credentials with env config
- Improved error handling with descriptive exceptions
