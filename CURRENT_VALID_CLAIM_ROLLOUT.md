# ERINYS Current Valid Claim / GraphRAG rollout

更新日: 2026-07-29

公開対象: ERINYS-mem v0.6.0

## 目的

ERINYSをGraphRAGへ全面置換しない。既存のFTS、Vector、RRFをbaselineとして維持し、関係探索が必要な質問だけに、時間・スコープ・競合・出典を管理した証拠検索経路を追加する。

ERINYSの役割は「GraphRAGより強い検索器」ではない。複数の検索方式の上で記憶の有効性と利用条件を管理するMemory Governance Layerとする。

## 変更しないもの

- 既存検索結果の順位と件数
- 既存MCPレスポンスの形
- 検索された回数だけで記憶を強化する処理
- 候補エッジをFactへ自動昇格する処理
- Graph DBへの移行

## Phase 0: Contract and fixed evaluation

状態: 完了

- `Current Valid Claim Evidence Path v0`のPydantic契約
- valid timeとrecorded timeの分離
- `ACTIVE / SUPERSEDED / CONTESTED / REFUTED / RETRACTED / ARCHIVED`
- 出典、変換、実行provenanceの分離
- 最大2-hop
- verified edgeだけを証拠検索経路へ採用
- outcome attributionのfail-closed判定
- 30ケースの固定fixture

出口条件:

- 固定30ケースが全件成功する
- ERINYS全体回帰が成功する

## Phase 1: Live read-only shadow

状態: 実装済み、既定OFF

現行の`rrf_hybrid_search`候補を複製し、Current Valid Claim評価へ渡す。返却結果は変更しない。

有効化:

```bash
ERINYS_CLAIM_SHADOW_MODE=audit
```

無効化・即時ロールバック:

```bash
ERINYS_CLAIM_SHADOW_MODE=off
```

`audit`時だけ、既存`audit_log`へ`operation=claim_shadow`の比較レシートを保存する。レシートに検索文、タイトル、本文は含めない。

保存する項目:

- query hashと文字数
- baseline candidate IDとscore
- graph candidate ID
- active、historical、conflicting claim ID
- baselineから除外・追加されたID
- selected / discarded path ID
- fallback理由
- evaluated / no_candidates / error状態とerror type
- adapter warning
- shadow評価時間

信頼境界:

- 既存`supersedes`は決定的なverified edgeとして扱う
- 既存`distilled_from`はverified derived edgeとして扱う
- その他の既存edgeは、`metadata.claim_edge`に検証状態がなければcandidate扱い
- candidate、rejected、retracted edgeは証拠検索経路へ入れない
- shadow評価失敗時はerror receiptとwarningを残し、既存検索を成功させる

集計:

```bash
python scripts/current_valid_claim_shadow_report.py --pretty
```

実DBコピーでのsoak:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/current_valid_claim_shadow_soak.py \
  --db /path/to/memory-copy.db \
  --retrieval-mode full_rrf_graph_copy \
  --query-source database \
  --count 100 \
  --warmup 5 \
  --pretty
```

`PYTHONPATH=src`は、本番と同じ拡張読込対応SQLiteを選ぶために必要。省略すると
標準SQLiteへ落ち、sqlite-vecを読めない環境がある。soak CLIは本番DB
`~/.erinys/memory.db`の指定を拒否する。

Phase 1のsoak出口条件:

- baselineレスポンス差分が0件
- shadow例外による検索失敗が0件
- レシートに生のquery、title、contentが0件
- receipt validation failureが0件
- shadow error rateを計測し、0件を目標にする
- 代表ワークロードでp95追加時間を計測し、許容値を決める
- fallback率、disagreement率、verified path率を最低100クエリで取得する

### 2026-07-27 copy-DB soak

対象は実DBの整合コピー44,752 observation。graph-linked observationのtitleから
100件を決定的に選び、生のqueryや本文をレシートへ保存せず、Vector + FTS + Graphの
完全RRF経路を測定した。このワークロードは一般検索の代表標本ではなく、graph fan-outを
意図的に強く踏むstress workloadである。

結果:

- baseline返却ID・順位の差分: 0 / 100
- 時刻依存score drift: 100 / 100、最大 `7.02e-10`
- valid receipt: 100 / 100
- shadow error / fallback: 0 / 100
- verified graph path: 100 / 100
- 追加時間: p50 `5.11 ms`、p95 `7.41 ms`
- shadow evaluator単体: p50 `4.64 ms`、p95 `4.90 ms`
- disagreement: 100 / 100、すべてgraph candidateの追加のみ
- baseline candidate: 常に50
- active candidate: median 128、max 128
- graph追加candidate: median 78、max 78
- selected path: median 64、max 64
- 本番DBの`claim_shadow` receipt: 0

初回計測ではnode上限外のedge endpointが残り、candidateが最大200まで膨張する欠陥を
検出した。edge採用時点で総node数128を強制するよう修正し、large graph回帰テストを
追加した。再計測ではmax 128を確認した。

判定:

- 非介入性、fail-open、receipt妥当性、遅延のPhase 1 safety gateは合格
- candidate/pathが上限に張り付くため、Phase 3の回答利用品質gateは不合格
- `ERINYS_CLAIM_SHADOW_MODE`は既定OFFを維持する
- fastembedのpooling既定変更警告があるため、embedding生成バージョンを固定し、
  既存vectorとの互換性を確認するまでcanaryへ進めない

## Phase 2: Claim and edge enrichment

状態: 開始条件確定、データ変更は未着手

shadowの実測結果を使い、暗黙推定を減らす。

- observation metadataへclaim status、confidence、scope、verification stateを明示
- edge metadataへrelation、origin、status、verification IDを明示
- evidence、source、claim、decision、outcomeのroleを分離
- 過去claimを削除せず、有効期間を閉じる
- migrationはdry-run、件数差分、rollback可能な形で行う

開始順:

1. read-only coverage reportで、明示claim metadataと明示verified edgeの率を測る
2. legacy `distilled_from`の全隣接展開をやめ、query intentとedge role別にbudgetを持つ
3. mixed workloadを追加し、lookupとgraph stressを分離して評価する
4. embedding model / pooling / implementation versionを固定する
5. dry-run差分を承認してから、初めてmetadata migrationを行う

出口条件:

- scope不明率とstatus不明率を測定できる
- verified path率がPhase 1より改善する
- 誤った自動昇格が0件
- representative mixed workloadでnode上限到達率が10%未満
- selected path上限到達率が10%未満
- graph追加candidateの中央値がbaseline candidateの50%以下
- selected pathに使うedgeは明示的なverification metadataを持つ
- embedding互換性試験が成功する

## Phase 3: Guarded canary retrieval

状態: 未着手

Current Valid Claim結果を回答候補として使うが、全質問へ適用しない。

対象候補:

- 現在有効な設定・契約・承認済みpolicy
- as-of質問
- 競合・反証の確認
- 出典経路の監査

対象外:

- 単純な詳細fact lookup
- creative評価
- 根拠のない原因推定
- graph evidenceが存在しない質問

出口条件:

- feature flagで即時baselineへ戻せる
- canary対象だけで回答差分を人手またはblind judge評価する
- citation correctness、scope correctness、temporal correctnessを個別採点する
- latency、fallback、誤除外に上限を置く

## Phase 4: Outcome-aware maintenance

状態: 未着手

検索・採用だけでは強化しない。

```text
retrieved
→ used
→ executed
→ verified
→ outcome observed
→ attribution accepted
→ reinforcement target selected
```

Outcome Attribution Receiptが揃った記憶だけを、指定された対象に限定して強化する。反証・撤回・安全上重要な履歴はDecayで消さない。

## 現在の判定

Phase 1の実装とcopy-DB safety soakは完了した。GraphRAGを回答経路へ有効化する段階ではない。
次はPhase 2Aのread-only coverageとfan-out抑制であり、metadata migrationやcanaryではない。
測定なしに本番回答をGraph経路へ切り替えない。
