"""Ask the corpus a question from the command line.

Retrieval only: prints the passages that match best, each as the publisher's
own text (texts.body, not the normalized copy that was embedded) under the
exact refs it came from. Nothing here writes an answer.

Results are grouped by citation. The same passage in several translations
(Sanhedrin 56a:24 in Davidson, the Community Translation, Goldschmidt's
German and the Aramaic) is one result: the best-matching version is quoted
and the others are listed under it. Versions chunk differently, so a hit
joins a group when any of its refs is already in that group.

Filters narrow the search before ranking, not after, so asking for Chabad
works cannot come back empty just because other works fill the top ranks.
A filter by work matters for attribution: "the Rebbe" in Sichot HaRan is
Rabbi Nachman of Breslov, and similarity search matches the word, not the
person.

Usage:
    python -m search.query "What are the seven Noahide laws?"
    python -m search.query "..." --category Chasidut --work Tanya --work "Torah Ohr"
    python -m search.query "..." --k 10 --lang en --model intfloat/multilingual-e5-base
"""
from __future__ import annotations

import argparse, json, sqlite3, textwrap

from embed.chunk import normalize
from embed.embed import fingerprint
from eval.score import DB_PATH, QUERY_PREFIX, load_vectors


def main(question: str, model: str, k: int, lang: str | None, categories: list[str],
         works: list[str], width: int) -> None:
    import torch
    from sentence_transformers import SentenceTransformer
    db = sqlite3.connect(DB_PATH)
    _, fp = fingerprint(db)
    cat = dict(db.execute("SELECT title, category FROM works WHERE base_work IS NULL"))
    rows = db.execute("SELECT work, lang, version_title, spans FROM chunks ORDER BY id").fetchall()

    keep = torch.ones(len(rows), dtype=torch.bool)
    for i, (work, c_lang, _, _) in enumerate(rows):
        if (lang and c_lang != lang) or \
           (categories and cat.get(work) not in categories) or \
           (works and not any(w.lower() in work.lower() for w in works)):
            keep[i] = False
    if not keep.any():
        raise SystemExit("no chunks match those filters")

    v = torch.from_numpy(load_vectors(model, fp)).cuda()
    enc = SentenceTransformer(model, device="cuda")
    qv = enc.encode([QUERY_PREFIX.get(model, "") + normalize(question)],
                    normalize_embeddings=True, convert_to_tensor=True)
    scores = (qv.to(v.dtype) @ v.T)[0].float().cpu()
    del v, enc; torch.cuda.empty_cache()
    scores[~keep] = float("-inf")
    order = scores.argsort(descending=True)[:int(keep.sum())].tolist()

    # group -> {(lang, version): (score, refs)}, best first. Only a group's
    # first hit claims refs; letting every joiner add its own would chain
    # neighbouring passages into one ever-growing group.
    groups: list[dict] = []
    owner: dict[str, int] = {}
    for cid in order[:50 * k]:      # deep enough to list the other versions
        work, c_lang, ver, spans = rows[cid]
        refs = list(dict.fromkeys(r for r, _, _ in json.loads(spans)))
        g = next((owner[r] for r in refs if r in owner), None)
        if g is None:
            if len(groups) == k:
                continue            # later hits may still join a shown group
            g = len(groups); groups.append({})
            for r in refs:
                owner[r] = g
        groups[g].setdefault((c_lang, ver), (scores[cid].item(), refs))

    for n, group in enumerate(groups, 1):
        hits = [(s, l, v, r) for (l, v), (s, r) in group.items()]
        score, c_lang, ver, refs = hits[0]
        head = refs[0] + (f" .. {refs[-1].rsplit(' ', 1)[-1]}" if refs[-1] != refs[0] else "")
        print(f"\n{n:>2}. {head}   score {score:.3f}")
        print(f"    [{c_lang}] {ver}")
        # The displayed text is the publisher's, segment by segment, so what
        # is quoted is exactly what the ref points at.
        for ref in refs:
            row = db.execute("SELECT body FROM texts WHERE ref = ? AND lang = ?"
                             " AND version_title = ?", (ref, c_lang, ver)).fetchone()
            if row and row[0]:
                body = row[0] if len(row[0]) <= 600 else row[0][:600] + " …"
                print(textwrap.fill(body, width, initial_indent="    ",
                                    subsequent_indent="    "))
        for score, c_lang, ver, _ in hits[1:]:
            print(f"      also [{c_lang}] {ver[:70]}  ({score:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="search.query")
    ap.add_argument("question")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--k", type=int, default=8, help="citations to show")
    ap.add_argument("--lang", help="only chunks in this language, e.g. en or he")
    ap.add_argument("--category", action="append", default=[],
                    help="only works in this category (repeatable): Chasidut, Talmud, ...")
    ap.add_argument("--work", action="append", default=[],
                    help="only works whose title contains this (repeatable)")
    ap.add_argument("--width", type=int, default=100)
    a = ap.parse_args()
    main(a.question, a.model, a.k, a.lang, a.category, a.work, a.width)
