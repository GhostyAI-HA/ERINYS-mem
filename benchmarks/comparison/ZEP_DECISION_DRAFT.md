# Zep (getzep) ベースライン包含判定ドラフト

- 対象: ERINYS論文（ARR 2026-10提出）ベースライン比較実験（`aionexo/paper/erinys/baseline_experiment_design.md` §1）
- 調査日: 2026-08-19（全URL同日確認）
- 判定期限: w2 (2026-08-29)

## 推奨: **exclude-with-reason（除外・理由明記）**

1. Zep本体のself-hosted版（Community Edition）は2025年4月に廃止済み。サポートされる実行形態はZep Cloud（SaaS）のみで、「ローカル実行」条件を満たす形のZepは現存しない。
2. OSSコアのGraphitiは動くが、それは「Zep」ではなく、論文で"Zep"として報告すると帰属が不正確になる（Zep Cloudのreranker/context assembly層を含まない）。
3. Graphitiで代替しても、session/document(episode)単位の検索はBM25のみでembeddingベースのepisode検索経路が確認できず、「埋め込みモデル統一で同一採点」という同条件要件をretrieval採点単位で満たせない。設計書の「正直な除外 > 雑な包含」に該当。

---

## 調査結果

### 1. ZepのOSS状態

- **Zep Community Edition (self-hosted) は廃止済み**。getzep/zep リポジトリのREADMEに「Zep Community Edition is no longer supported」と明記され、コードは `legacy/` フォルダへ移動。廃止時期は2025年4月。リポジトリ自体はアーカイブされていないが、現在はZep Cloud向けのexamples/integrations置き場になっている。
  - 一次: https://github.com/getzep/zep （README、2026-08-19確認）
  - 方針転換の公式ブログ: https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/
  - 旧CE発表: https://blog.getzep.com/announcing-zep-community-edition/
- ライセンス: OSSコアとして公式に提供されるのは **Graphiti (Apache-2.0)**。legacy/のCEコードのライセンスは今回未確認（廃止済みのため判定に影響しない）。
  - https://github.com/getzep/graphiti/blob/main/LICENSE

### 2. self-hosted版の実行可能性

- サポートされるself-hosted Zepは存在しない。旧CEのDocker構成は`legacy/`に残るが未保守で、2025年4月以降の修正なし。論文の再現性要件（読者が同条件で再実行できる）を満たさない。
- 現行の公式選択肢は (a) Zep Cloud（SaaS、ローカル実行不可）、(b) Graphitiを自前デプロイ、の2つ。
  - https://github.com/getzep/zep / https://help.getzep.com/faq

### 3. Graphiti（実質のOSSコア）

- **Pythonライブラリとして直接使用可**: PyPI `graphiti-core`（https://pypi.org/project/graphiti-core/）。Apache-2.0。
- **グラフDB依存（必須）**: Neo4j 5.26+ / FalkorDB 1.1.2+ / Amazon Neptune（Kuzuはdeprecated）。ローカルはNeo4jまたはFalkorDBのDocker起動が現実的。
  - https://github.com/getzep/graphiti （README）
- **LLM依存（ingest時必須）**: デフォルトOpenAI。Azure OpenAI / Gemini / Anthropic / Groq / Ollama（ローカル）に差し替え可。LLMなしの`add_episode`は不可（Issue #1299で要望が出ている段階）。設計書の「LLM使用はwrite時のみ許容」条件自体は満たせる。
  - https://github.com/getzep/graphiti/issues/1299
- **埋め込みモデル差し替え**: embedderは抽象化されており OpenAI / Voyage / Gemini / **SentenceTransformers** に対応。`pip install graphiti-core[sentence-transformers]` でbge-small-en-v1.5等のローカルモデルを指定可能。OpenAI text-embedding-3-smallも当然可。この点は条件を満たす。
  - https://github.com/getzep/graphiti/issues/437 / https://getzep-graphiti.mintlify.app/installation
- **ingest時のLLM呼び出し規模**: 1 episodeあたり複数回のLLM呼び出し（node抽出→dedup→edge抽出→per-edge解決・時刻付け・矛盾解消。dedupはMinHash/LSH高速パス+LLMフォールバック）。1 episodeあたり0.5〜数秒＋複数call。ベンチ全量（LoCoMo/ConvoMem/LongMemEvalの数千session）のingestはコスト・時間とも重い。
  - https://github.com/getzep/graphiti/issues/1193

### 4. retrieval評価（R@5/R@10, session/document単位）との適合性

- Graphitiの主検索単位は **edge（抽出済みfact）とnode（entity）**。ハイブリッド検索（embedding + BM25 + graph traversal + RRF）が効くのはこの層。
  - https://help.getzep.com/graphiti/working-with-data/searching
- **episode（=ingestした元会話単位）の検索は確認できた範囲でBM25のみ**（`EpisodeSearchMethod.bm25`、embeddingベースのepisode検索メソッドは一次ソース上で確認できず）。つまり:
  - (a) episode検索で採点 → 実質BM25ベースライン（既存の`baseline_bm25.py`と重複）。埋め込みモデル統一が採点対象経路に適用されない。
  - (b) edge/fact検索→edgeが持つepisode UUID群経由でsource episodeへ射影して採点 → 検索単位が他システムと非対称になり、射影・集約規則（複数edge→episodeのrank合成）が実験者の自由度になる。またfact自体がingest LLMの抽出品質に依存し、「pure retrieval比較」にならない。
- どちらの経路でも「全システム同一の採点スクリプトで同条件のR@k」という設計要件を素直に満たせない。

### includeした場合の工数参考（除外根拠の補強）

Mem0統合（OSS + ローカルqdrant）=2日相当に対し、Graphitiは **3〜5日+API費用** と見積もる: Neo4j/FalkorDB Docker整備0.5日、sentence-transformers embedder設定0.5日、全ベンチingest（数千episode×複数LLM call、レート制限込み）1〜2日、episode射影スコアラ設計・妥当性検証1〜2日。かつ最後の射影スコアラは上記(b)の非対称性を解消できず、投資しても「同条件」にならない。

---

## 論文本文用の除外理由ドラフト（英語）

> We exclude Zep from the locally-controlled comparison. Zep deprecated its self-hosted Community Edition in April 2025, leaving the managed Zep Cloud service as the only supported deployment, which is incompatible with our protocol of local execution with a unified embedding model. Its open-source core, Graphiti, performs semantic retrieval only over LLM-extracted nodes and edges while episode-level (document-level) search is BM25-only, so document-level R@k cannot be measured under the same conditions as the other systems.

（短縮版1文: We exclude Zep because its self-hosted edition was deprecated in April 2025 in favor of a cloud-only service, and its open-source core Graphiti does not support embedding-based document-level retrieval, precluding a same-conditions R@k comparison.）

---

### 要実機確認（もし方針転換でincludeする場合のみ）

- `graphiti_core/search/search_config.py` の `EpisodeSearchMethod` が本当にbm25のみか（本調査はdocs/統合コード経由の確認。pip install後にenumを直接確認するのが確実）
- `legacy/` CEのライセンス表記
