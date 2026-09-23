"""Split texts into retrieval chunks.

A Sefaria segment is the unit a citation points at, but it is a poor unit to
match against: 29% of Hebrew segments are under 15 words ("ואמר רבי יוחנן"),
too little meaning to find. So chunks are windows of consecutive segments,
and each chunk records which character ranges came from which segment --
retrieval matches the window, the answer cites the segment.

Rules, each for a reason:
  - One version per chunk. Merging Kehot and Sefaria translations would
    produce text neither published.
  - Never cross a section (chapter, daf, commented verse). A window that ends
    chapter 3 and begins chapter 4 matches neither well.
  - Reading order is texts.position, never segments.position -- the latter is
    set by whichever version landed first (107K collisions).
  - One Hebrew version per work. Some works carry up to 12 Hebrew editions
    (vocalized, unvocalized, reprints); embedding each fills the top results
    with copies of one passage. Every translation is kept: translations differ.
  - Nikkud and ta'amim are stripped from the embedded copy only. They split
    one word into many tokens and make identical words look different. The
    displayed text in `texts` is untouched.

Token budget uses the XLM-R tokenizer, which the candidate models share.

Usage:
    python -m embed.chunk run --scope cited      # Chasidut + what it cites
    python -m embed.chunk run --scope chasidut
    python -m embed.chunk run --scope all
"""
from __future__ import annotations

import argparse, json, re, sqlite3, time
from collections import defaultdict
from pathlib import Path

from store.migrate import migrate

ROOT       = Path(__file__).resolve().parent.parent
DB_PATH    = ROOT / "data" / "corpus.db"
TOKENIZER  = "intfloat/multilingual-e5-base"
MAX_TOKENS = 384        # e5 reads 512; leaves room for the "passage: " prefix

# Cantillation U+0591-05AF and vowel points U+05B0-05C7, minus maqaf (05BE,
# handled below) and sof pasuq (05C3, which is punctuation, not a mark).
MARKS = re.compile(r"[֑-ֽֿ-ׂׄ-ׇ]")
SECTION = re.compile(r"^(.*?)[ :](\d+[ab]?)$")
RANGE_END = re.compile(r"-[\d:ab]+$")


def normalize(s: str) -> str:
    s = MARKS.sub("", s.replace("־", " "))   # maqaf joins two words
    return " ".join(s.split())


def section(ref: str) -> str:
    """'Genesis 1:3' -> 'Genesis 1'; 'Berakhot 2a:5' -> 'Berakhot 2a';
    'Rashi on Genesis 1:1:2' -> 'Rashi on Genesis 1:1'. A ref with no trailing
    number is its own section."""
    m = SECTION.match(ref)
    return m.group(1) if m else ref


def cited_sections(db: sqlite3.Connection) -> set[str]:
    """Sections outside Chasidut that a Chasidut segment links to (either
    direction). A range cites from its start: 'Exodus 1:1-6:1' contributes
    'Exodus 1' only -- an undercount, never a wrong inclusion."""
    out = set()
    for a, b, ac in db.execute(
            "SELECT a_ref, b_ref, a_category FROM links"
            " WHERE (a_category = 'Chasidut') != (b_category = 'Chasidut')"):
        other = b if ac == "Chasidut" else a
        out.add(section(RANGE_END.sub("", other)))
    return out


def load(db: sqlite3.Connection, scope: str):
    """-> {(work, lang, version): [(position, ref, body), ...]} in scope."""
    cited = cited_sections(db) if scope == "cited" else None
    groups: dict[tuple, list] = defaultdict(list)
    q = ("SELECT s.work, s.category, t.lang, t.version_title, t.position, t.ref,"
         " t.body FROM texts t JOIN segments s ON s.ref = t.ref")
    if scope == "chasidut":
        q += " WHERE s.category = 'Chasidut'"
    for work, cat, lang, ver, pos, ref, body in db.execute(q):
        if cited is not None and cat != "Chasidut" and section(ref) not in cited:
            continue
        groups[(work, lang, ver)].append((pos, ref, body))
    for rows in groups.values():
        rows.sort()
    return groups


def one_hebrew_version(groups: dict) -> dict:
    """Keep the Hebrew version with the most segments per work (ties: words)."""
    best: dict[str, tuple] = {}
    for (work, lang, ver), rows in groups.items():
        if lang != "he":
            continue
        score = (len(rows), sum(len(b) for _, _, b in rows))
        if work not in best or score > best[work][0]:
            best[work] = (score, ver)
    return {k: v for k, v in groups.items()
            if k[1] != "he" or best[k[0]][1] == k[2]}


def split_long(text: str, offsets: list[tuple[int, int]], limit: int) -> list[str]:
    """Cut an over-long segment every `limit` tokens, backed up to a space.
    Backing up lets the next piece run a few tokens over `limit`; the model's
    512 leaves room for that."""
    pieces, start = [], 0
    for i in range(limit, len(offsets), limit):
        cut = text.rfind(" ", start + 1, offsets[i][0] + 1)
        if cut <= start:
            cut = offsets[i][0]
        pieces.append(text[start:cut]); start = cut
    pieces.append(text[start:])
    return [p.strip() for p in pieces if p.strip()]


def chunk_version(rows, tok, limit: int):
    """[(pos, ref, body)] of one version -> [(spans, text, tokens)]."""
    texts = [normalize(b) for _, _, b in rows]
    enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
    out, cur, cur_n, cur_sec = [], [], 0, None

    def flush():
        nonlocal cur, cur_n
        if cur:
            spans, parts, at = [], [], 0
            for ref, t in cur:
                spans.append([ref, at, at + len(t)]); parts.append(t); at += len(t) + 1
            out.append((spans, " ".join(parts), cur_n))
        cur, cur_n = [], 0

    for (_, ref, _), text, ids, offs in zip(rows, texts, enc["input_ids"],
                                            enc["offset_mapping"]):
        if not text:
            continue
        n, sec = len(ids), section(ref)
        if cur and (sec != cur_sec or cur_n + n > limit):
            flush()
        cur_sec = sec
        if n > limit:
            for piece in split_long(text, offs, limit):
                out.append(([[ref, 0, len(piece)]], piece,
                            len(tok(piece, add_special_tokens=False)["input_ids"])))
            continue
        cur.append((ref, text)); cur_n += n
    flush()
    return out


def run(scope: str, limit: int) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    migrate(DB_PATH)
    db = sqlite3.connect(DB_PATH)
    t0 = time.time()
    groups = one_hebrew_version(load(db, scope))
    print(f"{len(groups):,} versions in scope ({time.time()-t0:.0f}s)", flush=True)

    db.execute("DELETE FROM chunks")
    n = toks = 0
    by_lang: dict[str, int] = defaultdict(int)
    for (work, lang, ver), rows in sorted(groups.items()):
        batch = []
        for spans, text, k in chunk_version(rows, tok, limit):
            batch.append((n, work, lang, ver, spans[0][0], spans[-1][0],
                          json.dumps(spans, ensure_ascii=False), text, k))
            n += 1; toks += k; by_lang[lang] += 1
        db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)", batch)
    db.commit()
    print(f"{n:,} chunks, {toks:,} tokens, {time.time()-t0:.0f}s")
    for lang, k in sorted(by_lang.items(), key=lambda x: -x[1])[:8]:
        print(f"  {lang:4s} {k:>9,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="embed.chunk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--scope", choices=["chasidut", "cited", "all"], default="cited")
    r.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args()
    run(a.scope, a.max_tokens)
