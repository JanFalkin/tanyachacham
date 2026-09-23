"""Ingest Sefaria texts from the public GCS export bucket.

Why not the API: 6,462 works, each needing one call (plus one per node for
complex works like Tanya, which has 124) at a courtesy 0.35s = hours. The
bucket serves the same text as flat files with no rate limit, in parallel.

Why not merged.json: merging collapses versions and keeps only Sefaria's
SCRIPT bucket ("Hebrew"/"English"). 42 Yiddish versions are filed as Hebrew.
Per-version files carry `actualLanguage`, which is the real ISO code. The
Rebbe's sichos are Yiddish, so language is never inferred from script here.

Usage:
    python -m ingest.bucket plan
    python -m ingest.bucket run [--workers 12] [--category Chasidut] [--limit N]
"""
from __future__ import annotations

import argparse, json, queue, sqlite3, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from store.migrate import migrate

ROOT      = Path(__file__).resolve().parent.parent
DB_PATH   = ROOT / "data" / "corpus.db"
BOOKS_URL = ("https://raw.githubusercontent.com/Sefaria/Sefaria-Export/"
             "master/books.json")
BOOKS     = ROOT / "data" / "books.json"
UA        = "TanyaChaCham/0.1"

# Sefaria's `language` is a script bucket; `actualLanguage` is the real code.
SCRIPT = {"he": "hebrew", "yi": "hebrew", "arc": "hebrew", "jpa": "hebrew",
          "en": "latin", "de": "latin", "fr": "latin", "es": "latin",
          "pt": "latin", "it": "latin", "ru": "cyrillic"}


def load_books() -> list[dict]:
    if not BOOKS.exists():
        BOOKS.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(BOOKS_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r, open(BOOKS, "wb") as f:
            f.write(r.read())
    d = json.loads(BOOKS.read_text())
    return d["books"] if isinstance(d, dict) else d


def daf(i: int) -> str:
    """Talmud index -> daf notation: 1='1a', 2='1b', 3='2a'. Tractates start 2a."""
    return f"{(i + 1) // 2}{'a' if i % 2 else 'b'}"


def flatten(node, prefix, out, depth=0, talmud=False, nodes=None):
    """Nested text arrays -> (ref, string) leaves, reproducing canonical refs.

    `nodes`, if given, collects the node-qualified work titles passed through
    ("Tanya, Part I; Likkutei Amarim"). The link CSV names works that way and
    never by the base title, so those names have to become `works` rows for a
    citation to resolve back to "Tanya". Collecting them here rather than from
    `schema` keeps them identical to the refs by construction.
    """
    if isinstance(node, str):
        if node.strip():
            out.append((prefix, node))
    elif isinstance(node, list):
        for i, child in enumerate(node, start=1):
            if depth == 0:
                label = daf(i) if talmud else str(i)
                nxt = f"{prefix} {label}"
            else:
                nxt = f"{prefix}:{i}"
            flatten(child, nxt, out, depth + 1, talmud, nodes)
    elif isinstance(node, dict):
        # Complex work: keys are node titles, e.g. "Part I; Likkutei Amarim".
        # A blank key means an unnamed default node -- appending ", " there
        # yields "Likutei Moharan,  272:1" which matches nothing in the link
        # graph (canonical is "Likutei Moharan 272:1").
        for key, child in node.items():
            k = str(key).strip()
            nxt = f"{prefix}, {k}" if k else prefix
            if k and nodes is not None:
                nodes.append(nxt)
            flatten(child, nxt, out, 0, talmud, nodes)


def schema_nodes(schema, prefix="", he_prefix="", out=None):
    """Walk `schema` -> [(node title, Hebrew node title)].

    The text tree only shows nodes this version actually carries; the schema
    is the work's real structure and is the only place node Hebrew titles
    exist. Blank titles are skipped exactly as flatten() skips blank keys.
    """
    out = [] if out is None else out
    for n in (schema or {}).get("nodes") or []:
        en = str(n.get("enTitle") or "").strip()
        he = str(n.get("heTitle") or "").strip()
        title    = f"{prefix}, {en}" if (prefix and en) else (en or prefix)
        he_title = f"{he_prefix}, {he}" if (he_prefix and he) else (he or he_prefix)
        if en:
            out.append((title, he_title or None))
        schema_nodes(n, title, he_title, out)
    return out


TAGS = __import__("re").compile(r"<[^>]+>")
def clean(s: str) -> str:
    return TAGS.sub(" ", s).replace("&nbsp;", " ").strip()


def parse_version(doc: dict):
    """-> (lang, version title, category, leaves, work rows).

    The work rows are the base work followed by one row per node. Nodes carry
    `base_work` so a link naming "Tanya, Part I; Likkutei Amarim" resolves to
    "Tanya" with a join instead of a string heuristic.
    """
    title  = doc.get("title") or ""
    cats   = doc.get("categories") or []
    talmud = "Talmud" in cats
    # actualLanguage is authoritative; the bucket is only a fallback.
    lang   = (doc.get("actualLanguage")
              or {"he": "he", "en": "en"}.get(doc.get("language"), doc.get("language") or "??"))
    vtitle = doc.get("versionTitle") or "unknown"
    leaves: list[tuple[str, str]] = []
    nodes: list[str] = []
    flatten(doc.get("text"), title, leaves, talmud=talmud, nodes=nodes)

    cat      = cats[0] if cats else ""
    cat_path = "/" + "/".join(cats) if cats else ""
    # Schema nodes are the work's real structure and the only source of node
    # Hebrew titles; text nodes catch anything this version has that the
    # schema does not. Schema titles are relative, so qualify them with the
    # base title the way flatten() does.
    he_of = {f"{title}, {n}": he for n, he in schema_nodes(doc.get("schema"))}
    names = list(he_of) + [n for n in dict.fromkeys(nodes) if n not in he_of]
    complex_ = bool(names)

    works = [(title, doc.get("heTitle") or None, cat, cat_path, int(complex_), None)]
    works += [(n, he_of.get(n), cat, cat_path, 0, title) for n in names]
    return lang, vtitle, cat, leaves, works


def run(workers: int, category: str | None, limit: int) -> None:
    books = load_books()
    if category:
        books = [b for b in books if category in (b.get("categories") or [])]
    if limit:
        books = books[:limit]
    print(f"{len(books):,} version files to ingest "
          f"({len({b['title'] for b in books}):,} distinct works)", flush=True)

    migrate(DB_PATH)

    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-524288")

    writes: queue.Queue = queue.Queue(maxsize=200)
    stats = {"ok": 0, "fail": 0, "segs": 0, "words": 0}
    lock = threading.Lock()

    def writer():
        while True:
            item = writes.get()
            if item is None:
                writes.task_done(); return
            segs, txts, works = item
            # Versions of one work disagree about metadata -- an English
            # version has no heTitle, a partial one sees fewer nodes. Merge
            # rather than letting whichever landed last win.
            db.executemany(
                "INSERT INTO works(title,he_title,category,cat_path,is_complex,base_work)"
                " VALUES (?,?,?,?,?,?) ON CONFLICT(title) DO UPDATE SET"
                "   he_title   = COALESCE(works.he_title, excluded.he_title),"
                "   category   = COALESCE(NULLIF(works.category,''), excluded.category),"
                "   cat_path   = COALESCE(NULLIF(works.cat_path,''), excluded.cat_path),"
                "   is_complex = MAX(works.is_complex, excluded.is_complex),"
                "   base_work  = COALESCE(works.base_work, excluded.base_work)", works)
            db.executemany("INSERT OR IGNORE INTO segments(ref,work,category,position)"
                           " VALUES (?,?,?,?)", segs)
            db.executemany("INSERT OR REPLACE INTO texts"
                           "(ref,lang,script,version_title,source,body,words)"
                           " VALUES (?,?,?,?,?,?,?)", txts)
            db.commit()
            writes.task_done()

    wt = threading.Thread(target=writer, daemon=True); wt.start()

    def fetch(b: dict):
        try:
            req = urllib.request.Request(b["json_url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                doc = json.load(r)
            lang, vtitle, cat, leaves, works = parse_version(doc)
            work = b["title"]
            segs, txts, w = [], [], 0
            for pos, (ref, raw) in enumerate(leaves):
                body = clean(raw)
                if not body:
                    continue
                n = len(body.split()); w += n
                segs.append((ref, work, cat, pos))
                txts.append((ref, lang, SCRIPT.get(lang, "latin"), vtitle,
                             "sefaria", body, n))
            # Works go through even when this version carried no text, so a
            # work is never missing from the index because one version is empty.
            writes.put((segs, txts, works))
            with lock:
                stats["ok"] += 1; stats["segs"] += len(segs); stats["words"] += w
                if stats["ok"] % 500 == 0:
                    print(f"  {stats['ok']:>6,}/{len(books):,} versions  "
                          f"segs={stats['segs']:>9,}  words={stats['words']:>11,}  "
                          f"fail={stats['fail']}", flush=True)
        except Exception:
            with lock:
                stats["fail"] += 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fetch, books))
    writes.join(); writes.put(None); wt.join()

    db.execute(
        "INSERT INTO ingest_state(work,stage,segments,updated_at)"
        " SELECT work,'text',COUNT(*),datetime('now') FROM segments GROUP BY work"
        " ON CONFLICT(work) DO UPDATE SET stage=excluded.stage,"
        "   segments=excluded.segments, updated_at=excluded.updated_at")
    db.commit()

    print(f"\ndone in {time.time()-t0:.0f}s  ok={stats['ok']:,} fail={stats['fail']:,}", flush=True)
    w, c = db.execute("SELECT COUNT(*), SUM(is_complex) FROM works"
                      " WHERE base_work IS NULL").fetchone()
    print(f"  works={w:,} ({c or 0:,} complex)  "
          f"nodes={db.execute('SELECT COUNT(*) FROM works WHERE base_work IS NOT NULL').fetchone()[0]:,}")
    for lang, n, w in db.execute(
            "SELECT lang, COUNT(*), SUM(words) FROM texts GROUP BY lang ORDER BY 3 DESC"):
        print(f"  {lang:5s} rows={n:>9,}  words={w:>12,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="ingest.bucket")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--category")
    r.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "plan":
        bs = load_books()
        from collections import Counter
        print(f"versions: {len(bs):,} | works: {len({b['title'] for b in bs}):,}")
        for c, n in Counter((b.get('categories') or ['?'])[0] for b in bs).most_common():
            print(f"  {c:22s} {n:>7,}")
    else:
        run(a.workers, a.category, a.limit)
