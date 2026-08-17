# TASK PACK w2: Mem0アダプタ(スモーク+コストゲート) + LongMemEval-Sベースライン + ハーネス仕上げ

> 発注: HyperAION main loop (Claude) / 実装: codex gpt-5.5 luna max
> 前提: w1完了済み（commit ae18341）。このパックはその上に積む。
> cwd: /Users/fujiyoshi/oss/ERINYS-mem

## ゴール
1. comparison.csv スキーマに `top_k` 列を追加（run_comparison.py書き込み+make_table.py表示。既存rowはresults/内のsummary JSONのtop_kから復元してCSVを再構築するスクリプト or 手復元、方法は任せる。復元不能行があれば `unknown` と記す）
2. `benchmarks/comparison/README.md` 新設: --limitのセマンティクス（locomo=全体上限 / convomem=カテゴリ毎件数）、--top-k、再現コマンド一式（w1の8本+今回分）、生成物の命名規則、FINAL_REPORT/W1_RESULTS_TABLEへのポインタ
3. **LongMemEval-Sベースライン**: 既存 `benchmarks/longmemeval_bench.py` がデータをどこから読むか特定し、データセットを再取得（HF等の公式ソースから。~/.cache配下推奨）。comparison側にlongmemevalローダを追加し、rag/bm25 × k=5/10 の4本をフル計測（n=500想定）。ERINYSの再現値（99.4@5/100@10）との突合はしない（それはBENCHMARKS.md側の既存値）
4. **Mem0アダプタ（スモークのみ・コストゲート付き）**:
   - `.venv`に `mem0ai` を導入し、OSS/ローカルモードで `Mem0System` アダプタを実装（共通プロトコル準拠: ingest→search→ranked ids）
   - LLM設定は環境変数から（OPENAI_API_KEY等が無ければ**実行せずSKIPと報告**。キーの値をログ・ファイルに書かない）
   - **スモーク限定**: `--limit 5`（locomo）と `--limit 2`（convomem, カテゴリ毎2=10問）のみ実行可。フル計測は禁止
   - スモーク結果から**フル計測のコスト見積**を算出（LLM呼び出し回数×トークン概算→USD概算。モデルはmem0のデフォルト設定を記録）→ `MEM0_COST_ESTIMATE.md` に保存。フル実行はユーザー承認後の別パック

## 制約
- 変更可: `benchmarks/comparison/**` のみ（README.md新設含む）。他ディレクトリ・BENCHMARKS.md・src/は変更禁止
- ERINYS本体スコア・採点ロジックの変更禁止。共通採点コード(common.py)の採点部分は変更禁止（top_k列追加のような記録系はOK）
- git push禁止。commitはしない（作業ツリーに残す。コミットは発注側が検収後に行う）
- 外部ネットワーク: HFデータセット取得とmem0ai pip installのみ許可。Mem0のLLM呼び出しはスモーク範囲内のみ
- 数値の手打ち禁止: 全てrun_comparison.py実行出力から

## 検収条件（発注側が確認する）
- [ ] comparison.csvに`top_k`列があり、既存行も復元済み（または`unknown`明示）
- [ ] README.mdに--limitセマンティクス差が明記されている
- [ ] longmemeval: rag/bm25 × k5/k10 の4 summary JSONが存在し、n=500
- [ ] Mem0: スモーク2本のsummary JSON + MEM0_COST_ESTIMATE.md（フル未実行の明記）
- [ ] `git -C ~/oss/ERINYS-mem status`がcomparison配下のみの変更であること

## 完了報告
`benchmarks/comparison/W2_COMPLETION.md` に: 実行コマンド全列挙・生成ファイル一覧・LME 4本の数値表・Mem0スモーク数値とコスト見積・未解決事項。

## 発注側による前処理済み事項（2026-08-17 09:35追記）
- LongMemEval-S **raw** データ取得済み: `/tmp/longmemeval-data/longmemeval_s.json`（500問・HF xiaowu0162/longmemeval）
- **再現ギャップ発見**: BENCHMARKS.mdの再現コマンドが参照する `benchmarks/longmemeval_s_cleaned.json` はリポジトリにもdownload_datasets.shにも存在しない。→ README.mdの既知の問題節に記録せよ（BENCHMARKS.md自体は変更禁止のまま）
- 対応方針: LMEのbaseline計測は**raw版**で行い、同一データでの整合比較のため **ERINYS自身も raw版でlongmemeval_bench.pyを top-k 5/10 で再測定**せよ（結果はcomparison/results/へ。既存のBENCHMARKS.md値=cleaned版とは別物として明確にラベル）
- `mem0ai` は .venv に導入済み。OPENAI_API_KEY は環境に無い → Mem0スモークは実装+SKIP報告でよい（キー到着後に別途実行）
