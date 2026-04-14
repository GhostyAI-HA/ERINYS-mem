<p align="center">
  <img src="assets/logo.jpg" alt="ERINYS" width="200">
</p>

# ERINYS — AIエージェントのための反射記憶

[🇬🇧 English](README.md)

> **あった記憶から、なかった記憶すら生み出す。**

AIエージェントの記憶システムは、人間の記憶を模倣してきた。短期記憶、長期記憶、エピソード記憶、意味記憶。教科書通りの分類をそのまま実装に持ち込む。

使ってみて、ずっと違和感があった。

人間は忘れる。でも既存のメモリは忘れない。際限なく溜まり、古い情報が新しい情報と同じ重みで検索に出る。人間は「あれ、前と言ってること違くない？」と気づく。でもメモリは黙って上書きする。人間は2つの無関係な経験を結びつけて「あ、これ使える」と閃く。でもメモリはただ保存して返すだけ。

模倣すべきは記憶の分類じゃなかった。記憶の振る舞いだった。

その違和感がERINYSを呼んだ。

ERINYSは番犬だ。覚え、忘れ、疑い、噛む。

## 他のメモリシステムと何が違うか

**忘れる能力。** 多くのメモリは追記するだけで増え続ける。ERINYSはエビングハウスの忘却曲線で古い情報を沈め、アクセス頻度で重要な記憶を浮かせる。手動で整理しなくても検索結果が常に鮮度と関連性で最適化される。

**蒸留。** 具体的なバグ修正（「JWTのhttpOnlyフラグ漏れ」）から3層を自動生成する。事実→パターン（「新規エンドポイントにはセキュリティチェックリストが必要」）→原則（「セキュリティのデフォルトは安全側に倒すべき」）。プロジェクトを超えて使える教訓を記憶から抽出する。他のメモリシステムに蒸留機能はない。

**Dream Cycle。** 過去のメモ2つをLLMに渡して「関連はある？」と聞く。候補ペアは意味的類似度で選定。近すぎず（重複）遠すぎず（無関係）のスイートスポット（コサイン0.65–0.90）だけ。cronで夜間バッチ実行し、人間が並べて見ようとは思わない組み合わせから発見を生む。魔法ではなく、自動化されたノート比較。

## 設計思想

### 記憶には層がある

すべての記憶は同じではない。ERINYSは知識を抽象度で3層に分ける。

- **Concrete（具体）** — 起きた事実そのもの。「`/api/auth`エンドポイントでJWTのhttpOnlyフラグが抜けていた」
- **Abstract（抽象）** — 具体的事実から抽出されたパターン。「新しいAPIエンドポイントにはセキュリティヘッダーのチェックリストが必要」
- **Meta（原則）** — パターンから抽出された原則。「セキュリティのデフォルト値は、手動で有効化しなくても安全な状態にすべき」

1件のバグ修正から3層すべてが自動生成される。時間が経つと、Meta層にはプロジェクトも技術スタックも問わない普遍的な原則が蓄積される。

### 忘れることは機能である

すべての記憶には強度スコアがある。エビングハウスの忘却曲線に従い、時間とともに減衰する。半年前のメモは昨日のメモより低くランクされる。ただし頻繁にアクセスされた記憶は減衰に抵抗する。復習と同じ仕組み。

強度が閾値を下回ると、その記憶は刈り取り候補になる。DBが肥大化せず、検索結果のノイズが減る。

### 事実は変わる。履歴は消すな

情報が更新されたとき（「AWSからGCPに移行した」）、ERINYSは上書きしない。Supersede Chain（上書き連鎖）を作る。古い事実は「置き換え済み」とマークされるが消えない。「3月時点では何を信じていた？」と聞けば、その時点で有効だった答えが返る。

### 矛盾は検出すべき

エージェントの記憶に「PostgreSQLを使う」と「SQLiteを使う」が両方あったら矛盾。ERINYSはこれを検出して表面化させる。黙ってDBを切り替えるのではなく、「前にPostgreSQLを選んだけど、要件が変わった？」と聞く。

### 検索はキーワードだけでなく、意味も見つける

2つの検索を同時に実行し、結果を融合する。

- **キーワード検索**（FTS5）— 完全一致。高速で正確。
- **ベクトル検索**（sqlite-vec）— 意味の類似性。「認証」で検索すると「ログイン」「JWT」「セッショントークン」もヒットする。

2つの結果リストはRRF（Reciprocal Rank Fusion）で統合。両方で上位 = 最高スコア。

### すべてローカルで完結

SQLiteファイル1つ。クラウドAPI不要。APIキー不要。サブスク不要。オフライン動作。エージェントの記憶がマシン外に出ることはない。

## ユースケース

### 1. セッションをまたぐコーディングエージェントの記憶

```python
# バグ修正後にエージェントが学びを保存
erinys_save(
  title="JWT httpOnlyフラグ漏れを修正",
  content="CookieがJSからアクセス可能だった。httpOnly: true, secure: true, sameSite: strictを追加。",
  type="bugfix",
  project="my-app"
)

# 翌週、似たタスクが来たらエージェントが記憶を検索
erinys_search(query="認証 Cookie セキュリティ", project="my-app")
# → JWTの修正がスコア付きで返る。同じミスを繰り返さない。
```

### 2. 矛盾検出

```python
erinys_save(title="DB選定", content="シンプルさ重視でSQLiteを使用", project="my-app")
erinys_conflict_check(observation_id=42)
# → "⚠️ Observation #18 と矛盾: 'PostgreSQLを本番信頼性のために使用'"
```

### 3. Dream Cycle — 夜間の知識合成

```python
erinys_dream(max_collisions=10)
# ERINYSが「ちょうどいい距離」のペアを選ぶ（コサイン類似度 0.65–0.90）
# メモA: 「RTKでトークン消費を60-90%削減」
# メモB: 「Bootstrap Gateは5つのスクリプトを順次実行で3秒かかる」
# → 気づき: 「Bootstrap Gateの各スクリプトにRTKを適用すれば起動が速くなる」
```

### 4. 時間旅行クエリ

```python
erinys_timeline(query="デプロイ先", as_of="2026-03-01")
# → "AWS EC2 (2026-02-15に決定)"

erinys_timeline(query="デプロイ先", as_of="2026-04-01")
# → "GCP Cloud Run (2026-03-20にAWSから切り替え)"
```

### 5. 知識蒸留

```python
erinys_save(title="新エンドポイントにCORSヘッダー漏れ", type="bugfix", ...)
erinys_distill(observation_id=50, level="meta")
# → concrete: "/api/v2/usersエンドポイントにCORSヘッダーが未設定"
# → abstract: "新規APIエンドポイントにはCORSレビューチェックリストが必要"
# → meta:     "セキュリティ設定はデフォルトで安全にし、解除を明示的にすべき"
```

### 6. Obsidianエクスポート

```python
erinys_export(format="markdown")
# → [[wikilink]]付きの.mdファイルを生成
# Obsidianに入れればナレッジグラフが即座に可視化
```

## クイックスタート

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# MCPサーバーとして起動（stdio）
python -m erinys_memory.server

# テスト実行
PYTHONPATH=src pytest tests/ -v
```

## MCP設定

### Claude Desktop / Claude Code

```json
{
  "mcpServers": {
    "erinys": {
      "command": "/path/to/ERINYS-mem/.venv/bin/python3",
      "args": ["-m", "erinys_memory.server"],
      "env": {
        "ERINYS_DB_PATH": "~/.erinys/memory.db"
      }
    }
  }
}
```

### Antigravity

`~/.gemini/antigravity/settings.json` の `mcpServers` に追加:

```json
{
  "erinys": {
    "command": "/path/to/ERINYS-mem/.venv/bin/python3",
    "args": ["-m", "erinys_memory.server"],
    "env": {
      "ERINYS_DB_PATH": "~/.erinys/memory.db"
    }
  }
}
```

## 環境変数

| 変数 | デフォルト | 説明 |
|:--|:--|:--|
| `ERINYS_DB_PATH` | `~/.erinys/memory.db` | SQLiteデータベースのパス |
| `ERINYS_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed埋め込みモデル |

## Tools (25)

### Core
- `erinys_save` — Save observation (with topic_key upsert)
- `erinys_get` — Get by ID (full content, untruncated)
- `erinys_update` — Partial update
- `erinys_delete` — Delete with FK cascade
- `erinys_search` — RRF hybrid search (FTS5 + vector)
- `erinys_save_prompt` — Save user prompt
- `erinys_recall` — Recent observations
- `erinys_context` — Session context recall
- `erinys_export` — Obsidian-compatible markdown export
- `erinys_backup` — SQLite backup
- `erinys_stats` — Database statistics

### Graph
- `erinys_link` — Create typed edge
- `erinys_traverse` — BFS graph traversal
- `erinys_prune` — Prune weak/decayed edges

### Temporal
- `erinys_reinforce` — Boost observation strength
- `erinys_supersede` — Version an observation
- `erinys_timeline` — Query as-of timestamp
- `erinys_conflict_check` — Detect contradictions

### Dream Cycle
- `erinys_collide` — Collide two observations via LLM
- `erinys_dream` — Batch collision cycle

### Distillation
- `erinys_distill` — 3-granularity abstraction (concrete → abstract → meta)

### Batch & Eval
- `erinys_batch_save` — Bulk save with auto-linking
- `erinys_eval` — LOCOMO-inspired quality metrics

### Session
- `erinys_session_start` — Start session
- `erinys_session_end` — End session with summary
- `erinys_session_summary` — Save structured summary

## 他システムとの比較

| 機能 | ERINYS | Mem0 | Official MCP Memory |
|:--|:--|:--|:--|
| **ハイブリッド検索**（キーワード + ベクトル） | ✅ FTS5 + sqlite-vec RRF | ✅ Vector + graph | ❌ KGのみ |
| **忘却曲線** | ✅ エビングハウス | ⚠️ 優先度スコア | ❌ |
| **3段階蒸留**（具体 → 抽象 → 原則） | ✅ | ❌ | ❌ |
| **Dream Cycle**（記憶の衝突から洞察） | ✅ | ❌ | ❌ |
| **矛盾検出** | ✅ | ⚠️ Resolverで上書き | ❌ |
| **時間軸クエリ**（「3月時点で何を信じていた？」） | ✅ Supersede chain | ⚠️ グラフ無効化 | ❌ |
| **ローカル完結**（クラウドAPI不要） | ✅ SQLite 1ファイル | ❌ クラウド前提 | ✅ |
| **Obsidianエクスポート** | ✅ [[wikilinks]] | ❌ | ❌ |
| **Save時自動蒸留** | ✅ | ❌ | ❌ |
| **MCPネイティブ** | ✅ 25ツール | ✅ | ✅ |
| **自己評価**（LOCOMO指標） | ✅ | ❌ | ❌ |

> **要約** — 多くのメモリサーバーは保存と検索だけ。ERINYSはその上で忘れ、蒸留し、夢を見る。

## アーキテクチャ

```
┌──────────────────────────┐
│     FastMCP Server       │  25ツール、統一レスポンス形式
├──────────────────────────┤
│  search.py  │ graph.py   │  RRFハイブリッド │ 型付きエッジ
│  decay.py   │ session.py │  エビングハウス  │ ライフサイクル
│  temporal.py│collider.py │  バージョニング  │ 記憶の交差授粉
│  distill.py │ db.py      │  3段階蒸留      │ SQLite + vec
├──────────────────────────┤
│  embedding.py            │  fastembed (BAAI/bge-small-en-v1.5)
├──────────────────────────┤
│  SQLite + FTS5 + vec0    │  ローカル完結、ランタイムにネットワーク不要
└──────────────────────────┘
```

## ロードマップ

- [ ] Dream Daemon — Dream Cycleをバックグラウンドで自動実行
- [x] Save時の自動蒸留 — `erinys_save`のたびに3段階蒸留を自動実行
- [ ] 自動Prune — DB容量閾値超過時に減衰メモを自動刈り取り
- [ ] cron対応CLI — `erinys dream --max 10` で夜間バッチ実行
- [ ] PyPIパッケージ — `pip install erinys-memory`
- [ ] マルチエージェント — エージェントごとのスコープ分離

## ライセンス

MIT
