<p align="center">
  <img src="assets/logo.png" alt="ERINYS" width="600">
</p>

# ERINYS — AIエージェントのための検証可能なローカル記憶

<p>
  <a href="https://pypi.org/project/erinys-memory/"><img src="https://img.shields.io/pypi/v/erinys-memory" alt="PyPI"></a>
  <a href="https://github.com/GhostyAI-HA/ERINYS-mem/actions/workflows/test.yml"><img src="https://github.com/GhostyAI-HA/ERINYS-mem/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

**ローカル検索 約10ms。APIキー不要。トークン課金ゼロ。**

エージェント記憶のためのローカル信頼レイヤー。SQLiteファイル1つ、検索経路のLLM呼び出しはゼロ。検索リコールはLongMemEval-Sで99.4% R@5 / 100% R@10、より難しい`_m` splitで96.8% R@5 — 詳細は[ベンチマーク](#ベンチマーク)。

[🇬🇧 English](README.md) · [制約事項](docs/LIMITATIONS.md) · [比較](docs/COMPARISON.md) · [ツールリファレンス](docs/TOOLS.md) · [変更履歴](CHANGELOG.md)

---

AIエージェントの記憶システムは、人間の記憶を模倣してきた。短期記憶、長期記憶、エピソード記憶、意味記憶。教科書通りの分類をそのまま実装に持ち込む。

使ってみて、ずっと違和感があった。

人間は忘れる。でも既存のメモリは忘れない。際限なく溜まり、古い情報が新しい情報と同じ重みで検索に出る。人間は「あれ、前と言ってること違くない？」と気づく。でもメモリは黙って上書きする。人間は2つの無関係な経験を結びつけて「あ、これ使える」と閃く。でもメモリはただ保存して返すだけ。

模倣すべきは記憶の分類じゃなかった。記憶の振る舞いだった。その違和感がERINYSを呼んだ。

ERINYSは番犬だ。事実を保存し、履歴を残し、矛盾を捕まえ、削除を証明する。

> **出自:** ERINYSは[HyperAION](https://aionexo.com/hyperaion/)（AIエージェント自己改善フレームワーク）の検索レイヤーとして開発された。スタンドアロンのMCPサーバーとして公開しており、どのエージェントスタックからでも独立して利用できる。

## クイックスタート（30秒）

**1. インストール。**

```bash
pip install erinys-memory
```

**2. 環境を検証。** 1コマンドでPython・SQLite拡張サポート・sqlite-vec・埋め込み・依存・DBをチェックする。失敗した項目には `fix`（対処法）が出る。

```bash
erinys doctor
```

**3. MCPサーバーを登録。** クライアント（Claude Desktop / Claude Code / stdio対応のMCPホスト）に追加する。

```json
{
  "mcpServers": {
    "erinys": {
      "command": "erinys-memory",
      "env": {
        "ERINYS_DB_PATH": "~/.erinys/memory.db"
      }
    }
  }
}
```

**4. 保存して検索。** JSON CLIから実行（LLM不要・ネットワーク不要）。

```bash
erinys save --title "JWTのhttpOnlyフラグ漏れ" \
  --content "CookieがJSからアクセス可能だった。httpOnly, secure, sameSite=strictを追加。" \
  --type bugfix --project demo

erinys search "認証 Cookie セキュリティ" --project demo
```

検索はローカルのSQLiteファイル1つに対して約7〜10ms、LLM呼び出しゼロで走る。

## ベンチマーク

以下は**検索リコール**の数値（正しいセッションがTop-Kに入っているか）であり、エンドツーエンドのQA精度ではない — QA/回答可能性の評価ハーネスはv0.5.1で同梱、実測はこれから（[LIMITATIONS.md](docs/LIMITATIONS.md)）。全結果は同一モード（`enhanced_v2_boost`）、検索パイプラインの**LLM呼び出しゼロ**、現行依存関係での再現値（2026-07）。

| ベンチマーク | N | R@5 | R@10 |
|:--|:--|:--|:--|
| **LongMemEval-S** | 500 | **99.4%** | **100.0%** |
| **LongMemEval-M**（haystack約476セッション） | 500 | **96.8%** | 98.0% |
| **LoCoMo** | 1,982 | **92.7%**（公正値 ≈ 95.7%¹） | 97.2% |
| **ConvoMem** | 250 | 97.6%² | — |

¹ ミス監査の結果、LoCoMoのR@5ミスの42%はベンチマーク側のラベル欠陥（複数セッションに答えがあるのに単一gold、回答不能な敵対的質問）で、検索の失敗ではなかった。
² 2026年4月時点の構成。現行依存関係での再計測はこれから。

> **なぜ重要か：** APIキー不要。ネットワーク不要。検索にトークン消費ゼロ。ERINYSはFTS5 + sqlite-vec + アルゴリズムブーストだけでこの数値に到達している。エージェントの記憶検索はSQLiteの速度で動く。詳細な手法・カテゴリ別内訳・ミス分析・再現手順 → [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md)

ここに至るまでの物語 → [🇯🇵 日本語](docs/benchmark_story_ja.md) / [🇺🇸 English](docs/benchmark_story_en.md)

## 他のメモリシステムと何が違うか

**忘れる能力。** 多くのメモリは追記するだけで増え続ける。ERINYSはエビングハウスの忘却曲線で古い情報を沈め、アクセス頻度で重要な記憶を浮かせる。手動で整理しなくても検索結果の鮮度と関連性が保たれる。減衰は自動実行 — LLM不要。

**検証可能な忘却。** `erinys_forget`は記憶を*その派生物（蒸留された子孫）ごと*1トランザクションで削除し、全DB基盤（observations / ベクトル / FTS / エッジ / collisions）に残骸ゼロであることをmembershipテストで実証する。「削除をお願いする」のではなく「削除を証明する」。

**来歴（Provenance）。** すべての観測にはサーバーが刻印する来歴ブロックが付く — 誰が書いたか、どの書き込み経路か、どの親から派生したか。`erinys_lineage`で任意の記憶を起源まで遡れる。呼び出し側は偽装できない。

**蒸留。** 具体的なバグ修正（「JWTのhttpOnlyフラグ漏れ」）から3層を自動生成する。事実→パターン（「新規エンドポイントにはセキュリティチェックリストが必要」）→原則（「セキュリティのデフォルトは安全側に倒すべき」）。⚠️ *LLM呼び出しが必要（既定はローカルOllama）。*

**Dream Cycle。** 過去のメモ2つをLLMに渡して「関連はある？」と聞く。候補ペアは意味的類似度で選定 — 近すぎず（重複）遠すぎず（無関係）のスイートスポット（コサイン0.65–0.90）だけ。⚠️ *LLM呼び出しを使用 — ゼロLLM検索経路の外側で動く。*

> 蒸留とDream CycleはERINYSの生成的な一面だ。あった記憶から、なかった記憶すら合成する。どちらもLLMを使い、ゼロLLM検索の経路の外側で動く。

**適応検索。** クエリの複雑さを自動分類（L1/L2/L3）し、単純なキーワード検索はFTS寄り、複雑なマルチホップ質問はベクトル寄りに切り替える。WHY/WHO/WHEN の意図でブーストとグラフエッジ種別を調整し、グラフ到達可能な近傍を再ランキングする。CJKクエリは既定でベクトル検索に振られる — 非ラテン文字ではFTS5のporterトークナイザより埋め込みが強い。

## 設計思想

### 記憶には層がある

すべての記憶は同じではない。ERINYSは知識を抽象度で3層に分ける。

- **Concrete（具体）** — 起きた事実そのもの。「`/api/auth`エンドポイントでJWTのhttpOnlyフラグが抜けていた」
- **Abstract（抽象）** — 事実から抽出されたパターン。「新しいAPIエンドポイントにはセキュリティヘッダーのチェックリストが必要」
- **Meta（原則）** — パターンから抽出された原則。「セキュリティのデフォルト値は、手動で有効化しなくても安全な状態にすべき」

1件のバグ修正から蒸留で3層すべてが生成される。時間が経つと、Meta層にはプロジェクトも技術スタックも問わない普遍的な原則が蓄積される。

### 忘れることは機能である

すべての記憶には強度スコアがあり、時間とともに減衰する。半年前のメモは昨日のメモより低くランクされる。ただし頻繁にアクセスされた記憶は減衰に抵抗する — 復習と同じ仕組み。

強度が閾値を下回ると、その記憶は刈り取り候補になる。DBが肥大化せず、検索結果のノイズが減る。

### 事実は変わる。履歴は消すな

情報が更新されたとき（「AWSからGCPに移行した」）、ERINYSは上書きしない。Supersede Chain（上書き連鎖）を作る。古い事実は「置き換え済み」とマークされるが消えない。「3月時点では何を信じていた？」と聞けば、その時点で有効だった答えが返る。

### 矛盾は検出すべき

記憶に「PostgreSQLを使う」と「SQLiteを使う」が両方あったら矛盾。ERINYSはこれを検出して表面化させる。黙ってDBを切り替えるのではなく、「前にPostgreSQLを選んだけど、要件が変わった？」と聞く。

### 検索はキーワードだけでなく、意味も見つける

2つの検索を同時に実行し、結果を融合する。

- **キーワード検索**（FTS5）— NEARフレーズ展開つきの厳密一致。
- **ベクトル検索**（sqlite-vec）— ローカル埋め込みによる意味的類似。
- **RRF融合** — Reciprocal Rank Fusionが両ランキングを適応重みで統合。
- **意図ルーティング** — WHY/WHEN/WHOクエリでブーストとエッジ種別を調整。
- **グラフ再ランキング** — ナレッジグラフの近傍が融合スコアを押し上げる。

ループ内にLLMなし。検索レイテンシは15ms未満に収まる。

### すべてローカルで完結

SQLiteファイル1つ。クラウドAPI不要、サブスク不要、オフライン動作。エージェントの記憶がマシン外に出ることはない。

## ユースケース

**セッションをまたぐ記憶。** エージェントが学びを保存し、翌週のエージェントがそれを見つける。

```python
erinys_save(title="JWT httpOnlyフラグ漏れを修正",
            content="CookieがJSからアクセス可能だった。httpOnly, secure, sameSite=strictを追加。",
            type="bugfix", project="my-app")
erinys_search(query="認証 Cookie セキュリティ", project="my-app")
# → JWTの修正がスコア付きで返る。同じミスを繰り返さない。
```

**矛盾検出。**

```python
erinys_save(title="DB選定", content="シンプルさ重視でSQLiteを使用", project="my-app")
erinys_conflict_check(observation_id=42)
# → "⚠️ Observation #18 と矛盾: 'PostgreSQLを本番信頼性のために使用'"
```

**時間旅行クエリ。**

```python
erinys_timeline(query="デプロイ先", as_of="2026-03-01")  # → "AWS EC2 (2026-02-15に決定)"
erinys_timeline(query="デプロイ先", as_of="2026-04-01")  # → "GCP Cloud Run (2026-03-20にAWSから切り替え)"
```

**検証可能な忘却。**

```python
erinys_forget(id=42)                  # dry run: 削除される派生クロージャを表示
erinys_forget(id=42, dry_run=False)   # 削除して、全DB基盤に残骸ゼロを実証
```

## CLI

MCPサーバーはエージェント向けのアダプター。同じ操作はJSON CLIからも実行できる — 定期ジョブ・CI・復旧・手動検証向けに、安定したexit codeと機械可読な出力を返す。

```bash
erinys health --project my-app --deep --json     # 権威的ヘルスチェック: サーバーimport + 検索スモークテスト
erinys search "Buffer DNS" --project my-app --limit 5 --readonly --json
erinys context --project my-app --limit 10 --readonly --json
erinys save --title "決定" --content "What: ..." --type decision --project my-app --json
erinys undistilled --project my-app --limit 10 --json
erinys distill 123 --level meta --json
```

- `--readonly` はSQLite `mode=ro` で読む — キーワード検索のみ、マイグレーションや監査ログの書き込みなし。セマンティック検索が要るときは外す。
- `dream` / `prune` は**全プロジェクト横断**でDB全体に作用する。`prune --execute` はさらに `--confirm-global` が必要。
- 使い方エラーもJSONで返る（`error.code: "USAGE"`、exit code 2）。
- モジュール形式: `python -m erinys_memory.cli <command>`。

## MCPツール一覧（28ツール）

| 階層 | 数 | 内容 | LLM |
|:--|:--|:--|:--|
| **Basic** | 17 | save / search / recall / セッション管理 — 安定コア | ❌ なし |
| **Governance** | 7 | 来歴、supersede、timeline、矛盾検出、検証可能な忘却 | ❌ なし |
| **Experimental** | 4 | distill、dream、collide、eval — 研究的機能 | ⚠️ distill / dream / collide はLLMを呼ぶ |

全ツールが同じ `{ok, data, error}` エンベロープを返す。ツール別の完全なリファレンス → [docs/TOOLS.md](docs/TOOLS.md)

## 設定

| 変数 | デフォルト | 説明 |
|:--|:--|:--|
| `ERINYS_DB_PATH` | `~/.erinys/memory.db` | SQLiteデータベースのパス |
| `ERINYS_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | fastembed埋め込みモデル |
| `ERINYS_AUTO_DISTILL` | `1` | 保存時の自動蒸留（`0`で無効化） |
| `ERINYS_DISTILL_MODEL` | `gemma4:e4b` | 蒸留に使うローカルOllamaモデル |
| `ERINYS_DISTILL_ENDPOINT` | `http://localhost:11434/api/generate` | Ollama生成エンドポイント |

## アーキテクチャ

```
┌─────────────────────────────┐
│       FastMCP Server        │  28ツール、統一 {ok, data, error} エンベロープ
├─────────────────────────────┤
│ search.py     │ graph.py    │  RRFハイブリッド検索 │ 型付きエッジ
│ decay.py      │ session.py  │  エビングハウス減衰  │ ライフサイクル
│ temporal.py   │ collider.py │  バージョニング      │ Dream Cycle
│ distill.py    │ policy.py   │  3段階蒸留          │ アクセスポリシー
│ provenance.py │ db.py       │  来歴の刻印         │ SQLite + vec
├─────────────────────────────┤
│ embedding.py                │  fastembed (multilingual-MiniLM, 384d)
├─────────────────────────────┤
│ SQLite + FTS5 + sqlite-vec  │  完全ローカル、ランタイムにネットワーク不要
└─────────────────────────────┘
```

## 開発

```bash
git clone https://github.com/GhostyAI-HA/ERINYS-mem && cd ERINYS-mem
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v          # テスト実行
python -m erinys_memory.server           # ソースからMCPサーバーを起動（stdio）
ollama pull gemma4:e4b                   # distill / dream（LLM機能）を使う場合のみ
```

## リリースハイライト

- **v0.5** — ベンチマーク再現値の正直な報告（`_m` split初評価を含む）、時間グラウンディング、回答可能性、オプトインのメモリアクセスポリシー、QA/回答可能性の評価ハーネス
- **v0.4** — VMG: サーバー刻印の来歴、`erinys_lineage`、検証可能な忘却（`erinys_forget`）。MAGMA検索（適応重み・意図ルーティング・グラフ再ランキング）
- **v0.2** — 適応検索、意図認識ルーター、蒸留品質スコアリング、Dream結果スコアリング

詳細 → [CHANGELOG.md](CHANGELOG.md)

## ロードマップ

- [ ] Dream Daemon — Dream Cycleのバックグラウンド自動実行（現在は手動トリガー）
- [ ] DB容量閾値超過時の自動Prune
- [ ] マルチエージェント — エージェントごとのスコープ分離
- [ ] ConvoMem再計測 + 現行依存関係でのエンドツーエンドQA実測

## ライセンス

MIT © 2026 SHUN FUJIYOSHI (GhostyAI-HA) — [LICENSE](LICENSE)
