"""Score retrieval against eval/questions.jsonl.

For each question, embed it, rank every chunk by cosine similarity, and find
the first chunk that contains one of the question's `relevant` refs. A chunk
"contains" a ref when one of its spans cites it -- the same spans an answer
would cite, so a hit here is a citation that would check out.

  recall@k  share of questions with a relevant chunk in the top k
  MRR       mean of 1/rank of the first relevant chunk (0 if not in top 100)

Any language counts: an English question answered by the Hebrew chunk of the
right segment is a hit, because the citation is the same.

Usage:
    python -m eval.score                          # every model with vectors
    python -m eval.score --models BAAI/bge-m3 --show 5
"""
from __future__ import annotations

import argparse, glob, json, sqlite3
from pathlib import Path

import numpy as np

from embed.chunk import normalize
from embed.embed import fingerprint

ROOT      = Path(__file__).resolve().parent.parent
DB_PATH   = ROOT / "data" / "corpus.db"
VECTORS   = ROOT / "data" / "vectors"
QUESTIONS = ROOT / "eval" / "questions.jsonl"
KS        = (1, 5, 10, 20, 100)

# e5 was trained with role prefixes: "query: " for questions, "passage: " for
# the text searched. bge-m3 uses none.
QUERY_PREFIX = {"intfloat/multilingual-e5-small": "query: ",
                "intfloat/multilingual-e5-base":  "query: ",
                "intfloat/multilingual-e5-large": "query: "}


def load_vectors(model: str, fp: str) -> np.ndarray:
    d = VECTORS / model.replace("/", "__")
    meta = json.loads((d / "meta.json").read_text())
    # Vectors from an earlier chunking would pair with the wrong text and
    # report refs they never matched -- a false citation, silently.
    if meta["fingerprint"] != fp:
        raise SystemExit(f"{d} was built from different chunks -- re-embed it")
    v = np.concatenate([np.load(p) for p in sorted(glob.glob(str(d / "[0-9]*.npy")))])
    if len(v) != meta["chunks"]:
        raise SystemExit(f"{d}: {len(v):,} vectors for {meta['chunks']:,} chunks -- run incomplete")
    return v


def main(models: list[str], show: int) -> None:
    import torch
    from sentence_transformers import SentenceTransformer
    qs = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    db = sqlite3.connect(DB_PATH)
    _, fp = fingerprint(db)
    refs, label = [], []
    for spans, lang, ver in db.execute(
            "SELECT spans, lang, version_title FROM chunks ORDER BY id"):
        sp = json.loads(spans)
        refs.append({r for r, _, _ in sp})
        first, last = sp[0][0], sp[-1][0]
        label.append(f"[{lang}] {first}" + (f" .. {last.rsplit(' ', 1)[-1]}" if last != first else ""))
    print(f"{len(qs)} questions ({sum(q.get('status') == 'draft' for q in qs)} still draft)")

    for model in models:
        v = torch.from_numpy(load_vectors(model, fp)).cuda()
        enc = SentenceTransformer(model, device="cuda")
        qv = enc.encode([QUERY_PREFIX.get(model, "") + normalize(q["question"]) for q in qs],
                        normalize_embeddings=True, convert_to_tensor=True)
        top = (qv.to(v.dtype) @ v.T).topk(max(KS), dim=1).indices.cpu().numpy()
        del v, enc; torch.cuda.empty_cache()

        ranks = []
        print(f"\n=== {model}")
        for q, row in zip(qs, top):
            gold = set(q["relevant"])
            rank = next((i + 1 for i, c in enumerate(row) if refs[c] & gold), None)
            ranks.append(rank)
            print(f"  {q['id']}  rank {rank if rank else '>100':>4}  {q['question'][:60]}")
            for c in row[:show]:
                print(f"            {'*' if refs[c] & gold else ' '} {label[c]}")
        n = len(ranks)
        print("  " + "  ".join(f"recall@{k} {sum(1 for r in ranks if r and r <= k) / n:.2f}"
                               for k in KS)
              + f"  MRR {sum(1 / r for r in ranks if r) / n:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="eval.score")
    ap.add_argument("--models", nargs="+",
                    default=[json.loads((d / "meta.json").read_text())["model"]
                             for d in sorted(VECTORS.glob("*")) if not d.name.endswith("-smoke")
                             and (d / "meta.json").exists()])
    ap.add_argument("--show", type=int, default=3, help="top chunks to print per question")
    a = ap.parse_args()
    main(a.models, a.show)
