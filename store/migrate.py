"""Reconcile the live database with store/schema.sql.

WHY THIS EXISTS: `CREATE TABLE IF NOT EXISTS` is a no-op against a table that
already exists, even when the declaration has since gained a column. That is
silent -- no error, no warning -- so schema.sql and the database on disk drift
apart and the first symptom is `no such column` deep in a query. `base_work`
was declared on `works` and never reached the database that way.

The canonical shape is not parsed out of the SQL text. schema.sql is executed
into an in-memory database and the two are compared with PRAGMA table_info,
so whatever sqlite itself understands the schema to mean is what we compare.

SQLite's ALTER TABLE ADD COLUMN cannot add PRIMARY KEY or UNIQUE columns, and
cannot add NOT NULL without a default. Those are reported for a hand-written
rebuild rather than attempted.

Usage:
    python -m store.migrate            # apply
    python -m store.migrate --dry-run  # report only
"""
from __future__ import annotations

import argparse, sqlite3
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
DB     = ROOT / "data" / "corpus.db"
SCHEMA = ROOT / "store" / "schema.sql"


def shape(db: sqlite3.Connection) -> dict[str, dict[str, sqlite3.Row]]:
    """{table: {column: row}} as sqlite itself reports it."""
    db.row_factory = sqlite3.Row
    tables = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
        " AND name NOT LIKE 'sqlite_%'")]
    return {t: {r["name"]: r for r in db.execute(f"PRAGMA table_info({t})")}
            for t in tables}


def column_sql(col: sqlite3.Row) -> tuple[str, str | None]:
    """Replay one column as an ADD COLUMN clause, or explain why we can't."""
    if col["pk"]:
        return "", "is PRIMARY KEY"
    if col["notnull"] and col["dflt_value"] is None:
        return "", "is NOT NULL with no default"
    sql = f'"{col["name"]}" {col["type"] or "TEXT"}'
    if col["dflt_value"] is not None:
        sql += f" DEFAULT {col['dflt_value']}"
    return sql, None


def migrate(db_path: Path = DB, dry_run: bool = False) -> int:
    if not db_path.exists():
        print(f"{db_path} does not exist -- creating from schema.sql")
        db_path.parent.mkdir(parents=True, exist_ok=True)

    want = sqlite3.connect(":memory:")
    want.executescript(SCHEMA.read_text())
    canonical = shape(want)

    live = sqlite3.connect(db_path)
    # Creates anything wholly absent; existing tables are left untouched.
    live.executescript(SCHEMA.read_text())
    live.commit()
    have = shape(live)

    changes = 0
    for table, cols in canonical.items():
        missing = [c for name, c in cols.items() if name not in have.get(table, {})]
        for col in missing:
            sql, why = column_sql(col)
            if why:
                print(f"  ! {table}.{col['name']} {why}; needs a table rebuild "
                      f"-- not attempted")
                continue
            stmt = f"ALTER TABLE {table} ADD COLUMN {sql}"
            print(f"  + {table}.{col['name']:<12s} {'(dry run) ' if dry_run else ''}{stmt}")
            if not dry_run:
                live.execute(stmt)
            changes += 1

        extra = [n for n in have.get(table, {}) if n not in cols]
        for name in extra:
            print(f"  ? {table}.{name} exists in the database but not in "
                  f"schema.sql -- left alone")

    if not dry_run:
        live.commit()
    print(f"{changes} column(s) {'would be ' if dry_run else ''}added")
    return changes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="store.migrate")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    migrate(dry_run=a.dry_run)
