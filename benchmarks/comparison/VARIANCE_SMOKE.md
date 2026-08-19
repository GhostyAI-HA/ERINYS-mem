# Mem0 run-to-run variance smoke (N=3裁定, 2026-08-19)

目的: 設計書§2.5「LLM使用系はN=3」に対し、フル計測をN=1で確定した根拠を実測で与える。

## 方法

LoCoMo conversation 1 (`conv-26`, 197問) を**独立に3回ingest+検索**（store wipe済み・gpt-4.1-mini temp=0・conversation_sharedモード・全run同一197 case_id、検証済み）。

- run A = フル本走 (`results_mem0_locomo_full_20260819_131012.jsonl` の conv-26 スライス)
- run B = `summary_mem0_locomo_full_20260819_185706.json` (--limit 197, $0.084)
- run C = `summary_mem0_locomo_full_20260819_190007.json` (--limit 197, $0.084)

## 結果 (R@10, n=197)

| run | R@10 |
|:--|--:|
| A | 97.46 |
| B | 95.94 |
| C | 97.97 |

mean 97.12 / sd 1.06 / **spread 2.03pt**（1会話あたり。10会話のフルベンチでは独立変動が平均化され期待sd≈0.34pt）

## 裁定

- 温度0でもMem0のwrite時抽出は非決定（±2pt/会話）。
- 報告済みのERINYS−Mem0差（LoCoMo R@10 2.6pt / ConvoMem R@10 14.5pt）はこの変動を上回る。
- → フル計測N=1を採用し、分散をこのスモークで文書化する方式で§2.5を充足（設計書w2行の残債解消）。
- 論文への反映: main.tex §4.5 Baseline Comparison に1文（3値と判定を明記）。

総費用: $0.168（2走分。run Aはフル本走の再利用）
