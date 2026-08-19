# W2 completion — Mem0 LoCoMo full run (2026-08-19)

## Result

| System | Bench | Mode | Top-k | N | R@k % | Perfect % | p50 ms | p95 ms |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| mem0 | locomo | mem0_oss_local (gpt-4.1-mini extractor) | 10 | 1982 | 94.551 | 94.551 | 317.1 | 577.4 |

- 正本: `results/summary_mem0_locomo_full_20260819_131012.json` / `results/results_mem0_locomo_full_20260819_131012.jsonl`
- `results/comparison.csv` に追記済み（ConvoMem full 行もbackfill済み）
- カテゴリ別: adversarial 93.5 / open-domain 96.7 / single-hop 95.4 / temporal 94.7 / **temporal-inference 77.2**（最弱）
- 実費: 本走 $1.10（+初回セグメント $0.08 + プローブ2本 $0.25 ≒ 総額 $1.43）。ガバナー $40/$48、警報発火なし
- 参考（同一top_k=10・n=1982）: ERINYS 98.08 / BM25 94.70 / **Mem0 94.55** / naive RAG 58.27

## 走行中に直した3障害（コード修正はこのコミットに同梱）

1. **質問単位の会話再ingest** — LoCoMoは10会話×約200質問。per-caseモードでは会話全体を質問ごとに再抽出し54,916 LLM呼・約$166見込みだった。会話単位でストアを1回構築し全質問で共有する`conversation_shared`モードを実装（272呼・$1.10）。ERINYS/RAG/BM25ベンチと同じindex粒度になり公平条件も整合。
2. **TLS/connectタイムアウト即死** — `mem0_adapter._call_with_retry`（4回・バックオフ10/30/90s）をadd/searchに適用。
3. **`~/.mem0/migrations_qdrant`ロック** — mem0が内部に持つtelemetry/migrations用qdrantを`close()`対象に追加し、次会話の構築前にインスタンス参照を解放。

## 再現・監査メモ

- チェックポイント`--resume`を追加（per-case JSONL追記）。本走はconversation 1（197件・$0.08）をresumeで再利用した。summaryの`cost_scope=segment_only`はこのため。
- ingest前にケース/会話ストアをwipeするため、smoke残骸の二重ingestは以後起きない。
- **既知の注意（ConvoMem full 8/18）**: wipe実装前のため、smoke済み10/250ケースはストア既存のまま本走し、同一文書の二重ingestの可能性がある。recallへの影響は同一内容の重複であり限定的だが、論文掲載時に脚注で言及すること。
- `retrieval_llm_calls`フィールドはmem0行ではwrite時抽出呼数（LoCoMo=253）。検索自体はLLM無呼（ERINYSの"ゼロ"主張と同条件）。
