"""CLI entry point for chroma-memory-index."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import IndexConfig
from .core import get_client, get_stats, index_collection, query_collection
from .collector import collect_memory_files, collect_skill_files

logger = logging.getLogger("chroma_memory_index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chroma-memory-index",
        description="Chroma-backed semantic memory & skill index for AI agents (ARM64-safe)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # index
    idx = sub.add_parser("index", help="Index documents into ChromaDB")
    idx.add_argument("--memory", action="store_true", help="Index memory files")
    idx.add_argument("--skills", action="store_true", help="Index skill/tool files")
    idx.add_argument("--all", action="store_true", help="Index both memory and skills")
    idx.add_argument("--incremental", action="store_true", help="Only index new/changed docs")
    idx.add_argument("--full-rebuild", action="store_true", help="Force full rebuild")

    # query
    q = sub.add_parser("query", help="Search indexed documents")
    q.add_argument("text", help="Search query")
    q.add_argument(
        "--collection",
        choices=["memory", "skills", "all"],
        default="all",
        help="Collection to search (default: all)",
    )
    q.add_argument("-n", "--top", type=int, default=5, help="Number of results")

    # stats
    sub.add_parser("stats", help="Show collection statistics")

    # Global options
    parser.add_argument("--workspace", type=str, help="Workspace path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    return parser


def cmd_index(args: argparse.Namespace, config: IndexConfig) -> None:
    """Handle the 'index' subcommand."""
    client = get_client(config)

    do_memory = args.memory or getattr(args, "all", False)
    do_skills = args.skills or getattr(args, "all", False)

    if not do_memory and not do_skills:
        do_memory = do_skills = True  # default: index all

    incremental = args.incremental and not getattr(args, "full_rebuild", False)

    if do_memory:
        logger.info("indexing memory files...")
        docs = collect_memory_files(config)
        print(f"  Found {len(docs)} files")
        n = index_collection(client, config.memory_collection, docs, config, incremental)
        print(f"  ✅ Indexed {n} documents into '{config.memory_collection}'")

    if do_skills:
        logger.info("indexing skill/tool files...")
        docs = collect_skill_files(config)
        print(f"  Found {len(docs)} files")
        n = index_collection(client, config.skills_collection, docs, config, incremental)
        print(f"  ✅ Indexed {n} documents into '{config.skills_collection}'")


def cmd_query(args: argparse.Namespace, config: IndexConfig) -> None:
    """Handle the 'query' subcommand."""
    client = get_client(config)
    collections = []
    if args.collection in ("all", "memory"):
        collections.append(("memory", config.memory_collection))
    if args.collection in ("all", "skills"):
        collections.append(("skills", config.skills_collection))

    for label, col_name in collections:
        try:
            results = query_collection(client, col_name, args.text, config, args.top)
            print(f"\n🔍 {label} (top {args.top}):")
            for i, (doc, meta, dist) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                sim = 1 - dist
                print(f"  {i + 1}. [{sim:.2%}] {meta['source']}")
                print(f"     {doc[:150].strip()}...")
        except ValueError as e:
            print(f"\n🔍 {label}: {e}")
        except Exception as e:
            print(f"\n🔍 {label}: error — {e}")


def cmd_stats(args: argparse.Namespace, config: IndexConfig) -> None:
    """Handle the 'stats' subcommand."""
    client = get_client(config)
    stats = get_stats(client, [config.memory_collection, config.skills_collection])
    print("Collection statistics:")
    for name, count in stats.items():
        status = f"{count} documents" if count >= 0 else "not found"
        print(f"  {name}: {status}")


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Build config with CLI overrides
    config = IndexConfig.from_env()
    if args.workspace:
        from pathlib import Path
        from dataclasses import replace

        config = replace(config, workspace=Path(args.workspace))

    if args.command == "index":
        cmd_index(args, config)
    elif args.command == "query":
        cmd_query(args, config)
    elif args.command == "stats":
        cmd_stats(args, config)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
