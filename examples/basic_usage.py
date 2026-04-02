"""Example: basic indexing and querying."""
from chroma_memory_index.config import IndexConfig
from chroma_memory_index.core import get_client, get_stats, index_collection
from chroma_memory_index.collector import collect_memory_files

# Build config from environment
config = IndexConfig.from_env()

# Connect to ChromaDB
client = get_client(config)

# Collect and index memory files
docs = collect_memory_files(config)
print(f"Found {len(docs)} memory files")

# Index with incremental mode
n = index_collection(client, config.memory_collection, docs, config, incremental=True)
print(f"Indexed {n} documents")

# Check stats
stats = get_stats(client, [config.memory_collection, config.skills_collection])
for name, count in stats.items():
    print(f"  {name}: {count} docs")
