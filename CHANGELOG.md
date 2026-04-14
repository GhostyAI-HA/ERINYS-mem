# Changelog

## v0.1.0 (2026-04-14)

Initial release.

### Features

- 25 MCP tools via FastMCP (save, search, get, update, delete, recall, context, export, backup, stats)
- RRF hybrid search (FTS5 keyword + sqlite-vec vector similarity)
- Ebbinghaus forgetting curve decay with reinforcement
- Temporal versioning with supersede chains and timeline queries
- Contradiction detection across observations
- 3-granularity knowledge distillation (concrete → abstract → meta)
- Memory Collider engine with Dream Cycle batch processing
- Typed graph edges with BFS traversal
- Session lifecycle management
- Obsidian-compatible markdown export with [[wikilinks]]
- LOCOMO-inspired self-evaluation metrics
- Local-first: single SQLite file, no cloud APIs, no network at runtime
- fastembed ONNX embeddings (BAAI/bge-small-en-v1.5, 384-dim)
