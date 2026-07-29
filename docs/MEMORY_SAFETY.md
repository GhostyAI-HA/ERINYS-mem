# Admission / Activation Sentry

ERINYS separates provenance from authority. Provenance records where a memory
came from. Admission classifies what the stored record may be trusted for.
Activation decides whether a particular read may expose and use it.

## Admission

Every write path stamps server-controlled `metadata.memory_safety`:

```json
{
  "schema_version": 1,
  "source_class": "first_party_record",
  "trusted_for": ["context", "evidence"],
  "admission_verdict": "allow",
  "reasons": []
}
```

Source classes are `direct_user`, `first_party_agent`,
`first_party_record`, `policy`, `external`, `derived`, and `unknown`.
Caller metadata cannot forge this block. Importers may explicitly downgrade a
record to `external`; they cannot elevate a normal record to policy authority.

## Activation

Read operations declare one intended use:

- `context`: background material
- `evidence`: support for a claim
- `preference`: a user preference
- `directive`: authority to perform an action

`erinys_get`, `erinys_search`, `erinys_recall`, and `erinys_context` return an
activation receipt with `allow`, `context_only`, or `block`. Memory recalled
from a user, agent, archive, or external source does not become action authority
by retrieval alone. Directive use is deny-by-default unless ERINYS itself
recorded a policy source.

For untrusted records, instruction-like or prompt-injection content is replaced
with `[BLOCKED_BY_ERINYS_SENTRY]`. The receipt keeps a SHA-256 content hash for
audit without returning the blocked text.

## Legacy rows and compatibility

Sentry uses the existing metadata column and requires no historical rewrite.
Legacy rows are classified at read time from their source and provenance. The
v0.5.1 access policy remains a separate earlier gate: unauthorized candidates
are removed before claim-shadow evaluation or activation screening.

Sentry is a memory-use boundary, not an application permission system. The host
must still authenticate callers, authorize external actions, and obtain any
required human approval.
