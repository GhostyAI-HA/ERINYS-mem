# Mem0 cost estimate (W2)

Status: `SKIP` when the smoke summary is gated; no Mem0 LLM call is made in that state.

Settings (secrets omitted): `{"embedder_model": "text-embedding-3-small", "embedder_provider": "openai", "llm_model": "gpt-4.1-mini", "llm_provider": "openai", "package": "mem0ai", "required_env": "OPENAI_API_KEY", "required_env_present": true, "storage_dir": "/Users/fujiyoshi/oss/ERINYS-mem/benchmarks/comparison/results/mem0_local", "vector_store_mode": "local_oss", "vector_store_provider": "qdrant"}`

The projection is derived from the smoke summary and the full cached benchmark loader.
The adapter performs one `Memory.add(infer=True)` call per source document; input tokens use the recorded approximation when provider usage is unavailable.

| Benchmark | Smoke status | Smoke docs | Full cases | Full docs | Projected LLM calls | Projected input tokens | Projected output tokens | USD |
|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| locomo | COMPLETED | 19 | 1982 | 54916 | 54916 | 44831688 | 25353850 | UNAVAILABLE |
| convomem | COMPLETED | 226 | 36860 | 1505094 | 1505094 | 52292027 | 104437540 | UNAVAILABLE |

USD: `UNAVAILABLE` because the smoke was not executed and no provider pricing rates were supplied. Set `MEM0_INPUT_USD_PER_1M` and `MEM0_OUTPUT_USD_PER_1M` only after confirming the current model pricing.

Full Mem0 execution remains blocked pending explicit approval and credentials.
