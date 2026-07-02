"""P0-7 Memory access gate (app-level, zero-LLM, opt-in).

Similarity retrieval is a trust boundary: a caller scoped to project/namespace
A must not silently retrieve — or write into — memories that belong to B. This
module is the single, pure decision layer for that gate.

**What this is.** A deterministic, dependency-free policy over the plain dicts
that ``server.py`` already produces (observation payloads for writes, normalized
observation rows for reads). Two entry points:

- :func:`admit_memory` — should this write be persisted?
- :func:`retrieve_policy` — drop rows the principal may not see from a result set.

**What this is NOT.** This is authorization at the application boundary, not
cryptography. It filters rows *after* they were fetched from the local SQLite
DB; it does not encrypt them, does not stop a caller with direct DB/file access,
and does not authenticate the ``principal`` string (callers assert their own
identity). It is a guard rail against *accidental cross-scope leakage/pollution*
inside a trusted process, not a defense against a hostile local operator.

**Fail-closed within an active policy.** When a policy is configured (a non-None
``allowed_projects`` and/or ``allowed_scopes``) it is applied strictly: a row is
admitted only if it clears every constraint that is present. When *no* policy is
configured (all constraints ``None``), both functions are pass-through and
behavior is identical to pre-P0-7 ERINYS — that is what keeps the change
backward-compatible and opt-in.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "PolicyDecision",
    "policy_is_active",
    "normalize_scope_set",
    "extract_project",
    "extract_scope",
    "extract_principal",
    "admit_memory",
    "retrieve_policy",
]

# Scopes that are visible to any principal regardless of the project allow-list.
# ``global`` memories are shared knowledge by design; project restrictions only
# constrain project-scoped rows. Callers can still narrow this via
# ``allowed_scopes`` (e.g. deny ``personal`` entirely).
_PROJECT_AGNOSTIC_SCOPES = frozenset({"global"})

# A sentinel used nowhere externally; kept explicit so ``(ok, reason)`` tuples
# read the same across the two functions.
PolicyDecision = tuple[bool, str]


def policy_is_active(
    allowed_projects: Iterable[str] | None,
    allowed_scopes: Iterable[str] | None,
) -> bool:
    """True when at least one constraint is configured.

    ``None`` means "no constraint of this kind". An *empty* collection is a
    real (deny-everything) constraint and therefore counts as active — this is
    deliberate so ``allowed_projects=[]`` locks the caller out rather than
    silently disabling the gate.
    """
    return allowed_projects is not None or allowed_scopes is not None


def normalize_scope_set(scopes: Iterable[str] | None) -> frozenset[str] | None:
    """Return a lower-cased frozenset, or ``None`` when unconstrained."""
    if scopes is None:
        return None
    return frozenset(str(s).strip().lower() for s in scopes if str(s).strip())


def _normalize_project_set(projects: Iterable[str] | None) -> frozenset[str] | None:
    if projects is None:
        return None
    return frozenset(str(p).strip() for p in projects if str(p).strip())


def extract_project(row: Mapping[str, Any]) -> str | None:
    """Project a row/payload belongs to, or ``None`` (unscoped/global)."""
    project = row.get("project")
    if project is None:
        return None
    text = str(project).strip()
    return text or None


def extract_scope(row: Mapping[str, Any]) -> str:
    """Scope of a row/payload, defaulting to ``project`` (the DB default)."""
    scope = row.get("scope")
    if scope is None:
        return "project"
    text = str(scope).strip().lower()
    return text or "project"


def extract_principal(row: Mapping[str, Any]) -> str | None:
    """Principal that authored the row.

    Prefers ``metadata.provenance.principal`` (where ``server.py`` records it),
    then a top-level ``principal`` key (used by write payloads before the
    provenance block is materialized), else ``None``.
    """
    meta = row.get("metadata")
    if isinstance(meta, Mapping):
        prov = meta.get("provenance")
        if isinstance(prov, Mapping):
            principal = prov.get("principal")
            if isinstance(principal, str) and principal.strip():
                return principal.strip()
    top = row.get("principal")
    if isinstance(top, str) and top.strip():
        return top.strip()
    return None


def _project_allowed(
    project: str | None,
    scope: str,
    allowed_projects: frozenset[str] | None,
) -> bool:
    """Whether ``project`` clears the project allow-list for a given scope.

    Project-agnostic scopes (``global``) bypass the project check entirely —
    they are shared knowledge, not owned by one project. A row with no project
    is treated the same way (nothing to scope it to).
    """
    if allowed_projects is None:
        return True
    if scope in _PROJECT_AGNOSTIC_SCOPES:
        return True
    if project is None:
        return True
    return project in allowed_projects


def _scope_allowed(scope: str, allowed_scopes: frozenset[str] | None) -> bool:
    if allowed_scopes is None:
        return True
    return scope in allowed_scopes


def admit_memory(
    observation_meta: Mapping[str, Any],
    principal: str | None = None,
    allowed_scopes: Iterable[str] | None = None,
    allowed_projects: Iterable[str] | None = None,
) -> PolicyDecision:
    """Decide whether a write is permitted.

    Args:
        observation_meta: the write payload / observation dict. Only ``project``
            and ``scope`` are consulted; unknown keys are ignored.
        principal: the caller asserting the write (unused by the default rules,
            accepted so callers can pass it uniformly and so future rules — e.g.
            "principal X may only write scope personal" — extend without a
            signature change).
        allowed_scopes: scopes this caller may write. ``None`` = any scope.
        allowed_projects: projects this caller may write into. ``None`` = any
            project. A project-scoped write whose ``project`` is outside the set
            is blocked; ``global`` writes bypass the project check.

    Returns:
        ``(ok, reason)``. ``ok`` is False when a configured constraint is
        violated; ``reason`` is a short machine-stable code. When no constraint
        is configured, always ``(True, "no-policy")`` — writes are unchanged.
    """
    allowed_scope_set = normalize_scope_set(allowed_scopes)
    allowed_project_set = _normalize_project_set(allowed_projects)

    if not policy_is_active(allowed_projects, allowed_scopes):
        return True, "no-policy"

    scope = extract_scope(observation_meta)
    project = extract_project(observation_meta)

    if not _scope_allowed(scope, allowed_scope_set):
        return False, f"scope-denied:{scope}"
    if not _project_allowed(project, scope, allowed_project_set):
        return False, f"project-denied:{project}"
    return True, "admitted"


def _row_visible(
    row: Mapping[str, Any],
    allowed_project_set: frozenset[str] | None,
    allowed_scope_set: frozenset[str] | None,
) -> bool:
    scope = extract_scope(row)
    project = extract_project(row)
    if not _scope_allowed(scope, allowed_scope_set):
        return False
    if not _project_allowed(project, scope, allowed_project_set):
        return False
    return True


def retrieve_policy(
    results: Sequence[Mapping[str, Any]],
    principal: str | None = None,
    allowed_projects: Iterable[str] | None = None,
    allowed_scopes: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter retrieval results down to what ``principal`` may see.

    Args:
        results: rows produced by search / recall (each a mapping with at least
            ``project`` and ``scope``; order is preserved).
        principal: the caller (accepted for symmetry / future per-principal
            rules; the default rules are project/scope based).
        allowed_projects: projects the caller may read. ``None`` = all projects.
            ``global``-scoped rows are always visible regardless of this set.
        allowed_scopes: scopes the caller may read. ``None`` = all scopes.

    Returns:
        A new list of ``dict`` rows in the original order, dropping any row the
        caller may not see. When no constraint is configured, returns the rows
        unchanged (as dicts) — retrieval behavior is identical to pre-P0-7.
    """
    if not policy_is_active(allowed_projects, allowed_scopes):
        return [dict(row) for row in results]

    allowed_project_set = _normalize_project_set(allowed_projects)
    allowed_scope_set = normalize_scope_set(allowed_scopes)

    return [
        dict(row)
        for row in results
        if _row_visible(row, allowed_project_set, allowed_scope_set)
    ]
