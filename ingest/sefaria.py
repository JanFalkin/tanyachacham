"""Pull Sefaria texts + the intertextual link graph into a local SQLite store.

Sefaria refs are the universal address space. Everything -- Tanya, a daf of
Gemara, a pasuk -- is addressed the same way, and Sefaria has already resolved
the citations between them. We consume that rather than rebuilding it.

Usage:
    python -m ingest.sefaria init
    python -m ingest.sefaria works                  # catalog only
    python -m ingest.sefaria text  "Tanya"          # one work
    python -m ingest.sefaria links "Tanya"
    python -m ingest.sefaria bulk  --category Chasidut
"""
from __future__ import annotations

import argparse, json, re, sqlite3, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "corpus.db"
BASE    = "https://www.sefaria.org"
UA      = "TanyaChaChum/0.1 (corpus ingest; contact via project maintainer)"

# Sefaria asks for courtesy; this keeps us well under any sane rate limit.
SLEEP   = 0.35
TAGS    = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------- http

def api(path: str, retries: int = 4) -> object:
    url = BASE + urllib.parse.quote(path, safe="/?&=,;:%")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                raise
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def clean(s: object) -> str:
    return TAGS.sub(" ", s).replace("&nbsp;", " ").strip() if isinstance(s, str) else ""


def nwords(s: str) -> int:
    return len(s.split())


# ---------------------------------------------------------------- db

def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


def init(db: sqlite3.Connection) -> None:
    db.executescript((ROOT / "store" / "schema.sql").read_text())
    db.commit()


# ---------------------------------------------------------------- catalog

def walk_index(node, path=""):
    """Flatten Sefaria's nested category tree into (cat_path, work_dict)."""
    out = []
    if isinstance(node, list):
        for n in node:
            out += walk_index(n, path)
    elif isinstance(node, dict):
        if "category" in node and "contents" in node:
            p = f"{path}/{node['category']}"
            for n in node["contents"]:
                out += walk_index(n, p)
        elif "title" in node:
            out.append((path, node))
    return out


def load_works(db: sqlite3.Connection) -> int:
    idx = api("/api/index")
    rows = []
    for cat_path, w in walk_index(idx):
        top = cat_path.strip("/").split("/")[0] if cat_path else ""
        rows.append((w.get("title"), w.get("heTitle"), top, cat_path,
                     1 if w.get("isComplex") else 0))
    db.executemany(
        "INSERT OR REPLACE INTO works(title,he_title,category,cat_path,is_complex)"
        " VALUES (?,?,?,?,?)", rows)
    db.commit()
    return len(rows)


# ---------------------------------------------------------------- text

def daf(i: int) -> str:
    """Talmud section index -> daf notation. Sefaria indexes 1='1a', 2='1b',
    3='2a', ... Tractates begin at 2a, so indices 1-2 are empty placeholders."""
    return f"{(i + 1) // 2}{'a' if i % 2 else 'b'}"


def flatten(node, prefix, out, depth=0, talmud=False):
    """Walk Sefaria's nested text arrays, emitting (ref, string) leaves.

    A book nests chapters -> verses and joins index positions with ':'. A
    tractate nests dapim -> segments, where the top level is daf notation
    rather than an integer, so 'Berakhot' index 3 is 'Berakhot 2a'.
    """
    if isinstance(node, str):
        if node.strip():
            out.append((prefix, node))
    elif isinstance(node, list):
        for i, child in enumerate(node, start=1):
            if depth == 0:
                label = daf(i) if talmud else str(i)
                child_prefix = f"{prefix} {label}"
            else:
                child_prefix = f"{prefix}:{i}"
            flatten(child, child_prefix, out, depth + 1, talmud)


def shape_leaves(node, out):
    """Collect fetchable node titles from a complex work's shape tree.

    A leaf structural node is one whose 'chapters' is a count (or list of
    counts) rather than a list of further nodes; its 'title' is a valid ref.
    """
    if isinstance(node, list):
        for n in node:
            shape_leaves(n, out)
    elif isinstance(node, dict):
        ch = node.get("chapters")
        if isinstance(ch, list) and ch and isinstance(ch[0], dict):
            shape_leaves(ch, out)
        elif node.get("title"):
            out.append(node["title"])


def _pull(db, title, node_ref, work_title, talmud, rows_out):
    d = api(f"/api/texts/{node_ref}?context=0&pad=0&commentary=0")
    base = d.get("ref") or node_ref
    cat = d.get("primary_category") or ""
    en_leaves, he_leaves = [], []
    flatten(d.get("text"), base, en_leaves, talmud=talmud)
    flatten(d.get("he"),   base, he_leaves, talmud=talmud)
    he_by_ref = dict(he_leaves)
    merged = {r: (he_by_ref.get(r, ""), e) for r, e in en_leaves}
    for r, h in he_leaves:
        merged.setdefault(r, (h, ""))
    for ref, (he, en) in merged.items():
        he_c, en_c = clean(he), clean(en)
        if he_c or en_c:
            rows_out.append((ref, work_title, cat, he_c, en_c, 0, nwords(he_c), nwords(en_c)))


def ingest_text(db: sqlite3.Connection, title: str, verbose=True) -> int:
    meta = db.execute("SELECT category,is_complex FROM works WHERE title=?", (title,)).fetchone()
    category, is_complex = (meta or ("", 0))
    talmud = category == "Talmud"

    # /api/index does not report isComplex, so probe: a complex work rejects a
    # whole-work fetch and must be pulled node by node off its shape tree.
    rows: list = []
    complex_now = bool(is_complex)
    if not complex_now:
        try:
            _pull(db, title, title, title, talmud, rows)
        except urllib.error.HTTPError as e:
            if e.code not in (400, 404):
                raise
            complex_now, rows = True, []

    if complex_now:
        nodes: list[str] = []
        shape_leaves(api(f"/api/shape/{title}"), nodes)
        for n in nodes:
            try:
                _pull(db, title, n, title, talmud, rows)
            except Exception:
                pass
            time.sleep(SLEEP)
        db.execute("UPDATE works SET is_complex=1 WHERE title=?", (title,))

    rows = [(r[0], r[1], category or r[2], r[3], r[4], i, r[6], r[7])
            for i, r in enumerate(rows)]
    db.executemany(
        "INSERT OR REPLACE INTO segments"
        "(ref,work,category,he,en,position,he_words,en_words) VALUES (?,?,?,?,?,?,?,?)", rows)
    db.execute("INSERT OR REPLACE INTO ingest_state(work,stage,segments,updated_at)"
               " VALUES (?,?,?,datetime('now'))", (title, "text", len(rows)))
    db.commit()
    if verbose:
        hw = sum(r[6] for r in rows); ew = sum(r[7] for r in rows)
        print(f"  {title:38s} segments={len(rows):>6,}  he_words={hw:>9,}  en_words={ew:>9,}", flush=True)
    return len(rows)


def _dead(db, title, verbose=True):
    d = api(f"/api/texts/{title}?context=0&pad=0&commentary=0")
    base = d.get("ref") or title


# ---------------------------------------------------------------- links

def ingest_links(db: sqlite3.Connection, title: str, verbose=True) -> int:
    """Pull the citation graph for every segment of a work."""
    refs = [r[0] for r in db.execute(
        "SELECT ref FROM segments WHERE work=? ORDER BY position", (title,))]
    total = 0
    for i, ref in enumerate(refs):
        try:
            links = api(f"/api/links/{ref}")
        except Exception:
            continue
        rows = [(ref, l.get("ref"), l.get("category"), l.get("type"))
                for l in links if isinstance(l, dict) and l.get("ref")]
        if rows:
            db.executemany("INSERT OR IGNORE INTO links(a_ref,b_ref,category,link_type)"
                           " VALUES (?,?,?,?)", rows)
            total += len(rows)
        if i % 50 == 0:
            db.commit()
        time.sleep(SLEEP)
    db.execute("INSERT OR REPLACE INTO ingest_state(work,stage,segments,updated_at)"
               " VALUES (?,?,?,datetime('now'))", (title, "links", len(refs)))
    db.commit()
    if verbose:
        print(f"  {title:38s} links={total:,} over {len(refs):,} segments")
    return total


# ---------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser(prog="ingest.sefaria")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("works")
    p = sub.add_parser("text");  p.add_argument("title")
    p = sub.add_parser("links"); p.add_argument("title")
    p = sub.add_parser("bulk")
    p.add_argument("--category", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--with-links", action="store_true")
    p = sub.add_parser("stats")
    a = ap.parse_args()

    db = connect()
    if a.cmd == "init":
        init(db); print(f"initialised {DB_PATH}")
    elif a.cmd == "works":
        init(db); print(f"catalogued {load_works(db):,} works")
    elif a.cmd == "text":
        ingest_text(db, a.title)
    elif a.cmd == "links":
        ingest_links(db, a.title)
    elif a.cmd == "bulk":
        titles = [r[0] for r in db.execute(
            "SELECT title FROM works WHERE cat_path LIKE ? ORDER BY title",
            (f"%{a.category}%",))]
        if a.limit:
            titles = titles[:a.limit]
        print(f"{len(titles)} works in category matching {a.category!r}")
        for t in titles:
            done = db.execute("SELECT stage FROM ingest_state WHERE work=?", (t,)).fetchone()
            if done and done[0] in ("text", "links", "done"):
                print(f"  {t:38s} (already ingested, skipping)"); continue
            try:
                ingest_text(db, t)
            except Exception as e:
                print(f"  {t:38s} SKIP ({type(e).__name__}: {e})")
            time.sleep(SLEEP)
            if a.with_links:
                try: ingest_links(db, t)
                except Exception as e: print(f"  {t:38s} links SKIP ({e})")
    elif a.cmd == "stats":
        for label, q in [
            ("works catalogued", "SELECT COUNT(*) FROM works"),
            ("works ingested",   "SELECT COUNT(*) FROM ingest_state"),
            ("segments",         "SELECT COUNT(*) FROM segments"),
            ("links",            "SELECT COUNT(*) FROM links"),
            ("hebrew words",     "SELECT COALESCE(SUM(he_words),0) FROM segments"),
            ("english words",    "SELECT COALESCE(SUM(en_words),0) FROM segments"),
        ]:
            print(f"  {label:20s} {db.execute(q).fetchone()[0]:>12,}")

if __name__ == "__main__":
    main()
