# Timestamp-aware historical backfill

Backfill is the supported path for importing reviewed historical records into
ERINYS. It preserves when a record was created, when a claim was valid, and
when the record was demonstrably used. Direct SQLite inserts are unsupported:
they skip vector and FTS indexes, provenance, memory-safety classification,
idempotency, backups, and rollback bookkeeping.

## Time model

ERINYS keeps three clocks separate:

- `source.recorded_at`: when the source record was created
- `valid_from` / `valid_until`: when a claim was true
- `use_events[].occurred_at`: when there is evidence the record was used

The event ledger is authoritative. `created_at`, `updated_at`, `access_count`,
and `last_accessed` are search-compatible projections. Re-running the same
event keys cannot inflate access count or memory strength.

## JSONL contract

The CLI accepts one `BackfillItem` JSON object per non-empty line:

```json
{"topic_key":"refund-policy-2024","title":"Refund policy in 2024","content":"The limit was 50%.","project":"support","memory_role":"CLAIM","claim_status":"ACTIVE","source":{"kind":"slack","namespace":"slack:workspace-a","event_key":"sha256:record-1","recorded_at":"2024-01-10T09:00:00+09:00"},"valid_from":"2024-01-01T00:00:00+09:00","use_events":[{"event_key":"sha256:use-1","occurred_at":"2024-03-15T11:00:00+09:00"}]}
```

All timestamps require a timezone. One run may contain only one project, source
kind, and namespace. Unknown fields, duplicate event keys, future timestamps
beyond the configured clock skew, and caller-controlled trust metadata fail
closed. A `CLAIM` requires `claim_status`; a historical
`SUPERSEDED` / `REFUTED` / `RETRACTED` claim also requires `valid_until`.

Supported roles are `EVIDENCE`, `CLAIM`, `DECISION`, and `OUTCOME`. Backfill
does not infer edges from embedding similarity. It persists only supplied typed
relationships.

## Plan, apply, and verify

Apply is disabled by default. First identify trusted internal namespaces:

```bash
export ERINYS_FIRST_PARTY_RECORD_NAMESPACES="slack:workspace-a"
export ERINYS_BACKFILL_ENABLED=1
```

Generate a plan:

```bash
erinys backfill \
  --file slack-backfill.jsonl \
  --run-key slack-through-2024 \
  --project support \
  --json
```

Review counts, conflicts, temporal projections, and the returned
`plan_digest`. Apply the unchanged file with that exact digest:

```bash
erinys backfill \
  --file slack-backfill.jsonl \
  --run-key slack-through-2024 \
  --project support \
  --execute \
  --plan-digest sha256:... \
  --json
```

After apply, run the same command again and verify it reports an idempotent
result. Check `erinys health --project support --deep --json`, vector/FTS
membership, record counts, and representative searches before removing the
automatic pre-apply backup.

## Slack adapter

Convert reviewed Slack records before planning:

```bash
python scripts/backfill_slack.py \
  --input normalized-slack.jsonl \
  --output slack-backfill.jsonl
```

The adapter hashes workspace, channel, and message identity and redacts common
secrets, email addresses, and Slack mentions before embedding. Set
`ERINYS_BACKFILL_EVENT_HMAC_KEY` to make identifiers dependent on a local
secret; otherwise the adapter uses stable SHA-256.

Only namespaces listed in `ERINYS_FIRST_PARTY_RECORD_NAMESPACES` are classified
as `FIRST_PARTY_RECORD`. Other records may still be imported for context, but
cannot create verified edges, supersede an existing claim, or authorize an
action.

## Rollback

Rollback is local CLI-only and defaults to dry-run:

```bash
erinys backfill-rollback bf_... --json
erinys backfill-rollback bf_... --execute --json
```

Matched existing observations have their prior temporal projection restored.
Rows created by the run are removed only when no later event, external edge,
derived observation, or collision depends on them. Unsafe removal fails
closed; restore the automatic SQLite backup when full database rollback is
required.

## Operational limits

| Variable | Default | Purpose |
|:--|:--|:--|
| `ERINYS_BACKFILL_ENABLED` | `0` | Apply gate; planning stays available |
| `ERINYS_BACKFILL_MAX_ITEMS` | `100` | Local CLI cap; contract ceiling 1,000 |
| `ERINYS_BACKFILL_CHUNK_SIZE` | `100` | Observations per transaction |
| `ERINYS_BACKFILL_CLOCK_SKEW_SECONDS` | `300` | Future timestamp tolerance |
| `ERINYS_FIRST_PARTY_RECORD_NAMESPACES` | empty | First-party record allowlist |

The MCP surface remains hard-capped at 100 items per request. Use the local CLI
for large controlled migrations and split source data into bounded, resumable
runs.
