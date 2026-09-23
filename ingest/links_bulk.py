"""Bulk-load Sefaria's intertextual citation graph from the public GCS export.

Per-segment /api/links/ calls are ~0.35s each; across a million-plus segments
that is 100+ hours. Sefaria publishes the same graph as CSV shards in a public
bucket -- ~700MB, ~4-5M edges -- which loads in minutes instead.

    gs://sefaria-export/links/links0.csv ... links16.csv

Usage:
    python -m ingest.links_bulk fetch          # download shards (resumable)
    python -m ingest.links_bulk load           # parse + insert into corpus.db
    python -m ingest.links_bulk stats
"""
from __future__ import annotations

import argparse, csv, json, sqlite3, sys, time, urllib.request
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DB_PATH  = ROOT / "data" / "corpus.db"
LINK_DIR = ROOT / "data" / "links"
BUCKET   = "https://storage.googleapis.com/sefaria-export"
LIST_API = ("https://storage.googleapis.com/storage/v1/b/sefaria-export/o"
            "?prefix=links%2F&maxResults=200")
UA       = "TanyaChaChum/0.1"

csv.field_size_limit(10_000_000)


def shard_manifest() -> list[tuple[str, int]]:
    """List the links shards and their sizes, so we can verify downloads."""
    req = urllib.request.Request(LIST_API, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        items = json.load(r).get("items", [])
    # links_by_book*.csv are pre-aggregated rollups, not the edge list.
    return sorted(
        (i["name"], int(i["size"])) for i in items
        if i["name"].endswith(".csv") and "by_book" not in i["name"])


def fetch(force: bool = False) -> None:
    LINK_DIR.mkdir(parents=True, exist_ok=True)
    manifest = shard_manifest()
    total = sum(s for _, s in manifest)
    print(f"{len(manifest)} shards, {total/1e9:.2f} GB total")
    done = 0
    for name, size in manifest:
        dest = LINK_DIR / Path(name).name
        if dest.exists() and dest.stat().st_size == size and not force:
            print(f"  {dest.name:18s} cached  ({size/1e6:.0f} MB)", flush=True)
            done += size
            continue
        t0 = time.time()
        req = urllib.request.Request(f"{BUCKET}/{name}", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        got = dest.stat().st_size
        done += got
        ok = "ok" if got == size else f"SIZE MISMATCH exp={size:,} got={got:,}"
        print(f"  {dest.name:18s} {got/1e6:>6.0f} MB in {time.time()-t0:>5.1f}s  {ok}"
              f"   [{done/total*100:.0f}%]", flush=True)


def load() -> None:
    files = sorted(LINK_DIR.glob("links*.csv"),
                   key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    files = [f for f in files if "by_book" not in f.name]
    if not files:
        sys.exit("no shards on disk -- run `fetch` first")

    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=MEMORY")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-524288")          # ~512MB page cache
    # Indexes make a multi-million-row insert crawl; rebuild them afterwards.
    for idx in ("idx_link_a", "idx_link_b", "idx_link_aw", "idx_link_bw"):
        db.execute(f"DROP INDEX IF EXISTS {idx}")
    db.commit()

    SQL = ("INSERT OR IGNORE INTO links"
           "(a_ref,b_ref,a_work,b_work,a_category,b_category,link_type)"
           " VALUES (?,?,?,?,?,?,?)")
    grand, t_start = 0, time.time()

    for path in files:
        t0, n, batch = time.time(), 0, []
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rdr = csv.DictReader(fh)
            # Sefaria's header misspells it "Conection Type"; accept either.
            for row in rdr:
                a, b = row.get("Citation 1"), row.get("Citation 2")
                if not a or not b:
                    continue
                batch.append((
                    a, b,
                    row.get("Text 1"), row.get("Text 2"),
                    row.get("Category 1"), row.get("Category 2"),
                    row.get("Conection Type") or row.get("Connection Type") or "",
                ))
                if len(batch) >= 100_000:
                    db.executemany(SQL, batch); db.commit()
                    n += len(batch); batch.clear()
        if batch:
            db.executemany(SQL, batch); db.commit(); n += len(batch)
        grand += n
        print(f"  {path.name:18s} {n:>9,} rows in {time.time()-t0:>5.1f}s   (cum {grand:,})", flush=True)

    print("rebuilding indexes ...")
    t0 = time.time()
    db.executescript((ROOT / "store" / "schema.sql").read_text())
    db.execute("PRAGMA synchronous=NORMAL")
    db.commit()
    stored = db.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    print(f"indexes rebuilt in {time.time()-t0:.1f}s")
    print(f"parsed {grand:,} rows -> {stored:,} unique edges "
          f"in {time.time()-t_start:.0f}s total")


def stats() -> None:
    db = sqlite3.connect(DB_PATH)
    print(f"  edges          {db.execute('SELECT COUNT(*) FROM links').fetchone()[0]:>12,}")
    print("  by target category:")
    for cat, n in db.execute(
            "SELECT b_category, COUNT(*) c FROM links GROUP BY b_category"
            " ORDER BY c DESC LIMIT 12"):
        print(f"    {str(cat or '(none)'):24s} {n:>12,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="ingest.links_bulk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--force", action="store_true")
    sub.add_parser("load"); sub.add_parser("stats")
    a = ap.parse_args()
    {"fetch": lambda: fetch(a.force), "load": load, "stats": stats}[a.cmd]()
