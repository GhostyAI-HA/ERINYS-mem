"""Central sqlite3 implementation selector.

sqlite-vec (ERINYS's vector index) needs ``Connection.enable_load_extension``.
CPython ships that method disabled on some builds — macOS system Python and
pyenv built without ``--enable-loadable-sqlite-extensions`` are the common
offenders, while python.org / Homebrew / conda / GitHub Actions builds all
enable it.

**Import ``sqlite3`` from this module** (``from ._sqlite import sqlite3``) rather
than the standard library directly. Doing so guarantees every module shares one
implementation, so exception classes (``IntegrityError``, ``OperationalError``,
...) are identical across the codebase. If db.py used ``pysqlite3`` while other
modules caught ``stdlib sqlite3`` errors, those ``except`` clauses would silently
miss real failures.

Selection order:
  1. stdlib ``sqlite3`` when it can load extensions (the common case);
  2. ``pysqlite3`` if installed (fallback for extension-disabled builds);
  3. stdlib ``sqlite3`` otherwise — a clear, actionable error is then raised at
     connection time in ``db._load_sqlite_vec`` rather than a cryptic
     ``AttributeError``.
"""

from __future__ import annotations

import sqlite3 as _stdlib_sqlite3


def _select_sqlite3():
    try:
        probe = _stdlib_sqlite3.connect(":memory:")
        try:
            capable = hasattr(probe, "enable_load_extension")
        finally:
            probe.close()
    except Exception:
        capable = False
    if capable:
        return _stdlib_sqlite3
    try:
        import pysqlite3.dbapi2 as _pysqlite3  # type: ignore

        return _pysqlite3
    except ImportError:
        return _stdlib_sqlite3


sqlite3 = _select_sqlite3()

__all__ = ["sqlite3"]
