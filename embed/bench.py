"""Measure embedding throughput on this GPU, on our own chunks.

Published speeds are for other hardware and other text; Hebrew tokenizes
very differently from English. So: embed a random sample of real chunks,
time it, and project to every scope from the token count `embed.chunk`
recorded. This decides nothing about quality -- only the eval set does.

Usage:
    python -m embed.bench [--n 2000] [--models intfloat/multilingual-e5-base BAAI/bge-m3]
                          [--dtypes fp32 fp16]

With both dtypes, fp16 is also checked against fp32 on the same sample: the
cosine between each chunk's two vectors, and how many of each chunk's top-10
neighbours in the sample survive. Speed that reorders the neighbours is not
worth having -- the ranking is what picks the citation.
"""
from __future__ import annotations

import argparse, sqlite3, time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "corpus.db"
MODELS  = ["intfloat/multilingual-e5-base", "BAAI/bge-m3"]

# e5 was trained with role prefixes and degrades without them; bge-m3 uses none.
PREFIX = {"intfloat/multilingual-e5-small": "passage: ",
          "intfloat/multilingual-e5-base":  "passage: ",
          "intfloat/multilingual-e5-large": "passage: "}


def link_state() -> str:
    """The dock's PCIe link as the kernel reports it, for whichever NVIDIA card."""
    for d in Path("/sys/bus/pci/devices").iterdir():
        try:
            if (d / "vendor").read_text().strip() == "0x10de" and \
               (d / "class").read_text().startswith("0x03"):
                return (f"{(d/'current_link_speed').read_text().strip()} "
                        f"x{(d/'current_link_width').read_text().strip()}")
        except OSError:
            pass
    return "?"


def load(name: str, dtype: str):
    """A SentenceTransformer on the GPU, in fp32 or fp16. Pascal cards have no
    usable fp16; Ada (4060) and later get most of their speed from it."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name, device="cuda")
    model.max_seq_length = 512
    return model.half() if dtype == "fp16" else model


def bench(name: str, texts: list[str], batch: int, dtype: str = "fp32"):
    """-> (tokens/sec, seconds, peak VRAM GB, float32 vectors)."""
    import torch
    model = load(name, dtype)
    texts = [PREFIX.get(name, "") + t for t in texts]
    ntok = sum(len(x) for x in model.tokenizer(texts, truncation=True,
                                                max_length=512)["input_ids"])
    model.encode(texts[:batch], batch_size=batch)           # warm-up, CUDA init
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    vecs = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    torch.cuda.synchronize()
    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"    link under load: {link_state()}")
    del model; torch.cuda.empty_cache()
    return ntok / dt, dt, vram, vecs


def agreement(a, b, k: int = 10) -> tuple[float, float]:
    """-> (min cosine between paired vectors, mean top-k neighbour overlap)."""
    import numpy as np
    cos = (a * b).sum(1)
    top = lambda v: np.argsort(-(v @ v.T), axis=1)[:, 1:k + 1]
    ta, tb = top(a), top(b)
    overlap = np.mean([len(set(x) & set(y)) / k for x, y in zip(ta, tb)])
    return float(cos.min()), float(overlap)


def main(n: int, models: list[str], batch: int, dtypes: list[str]) -> None:
    import torch
    db = sqlite3.connect(DB_PATH)
    total, toks = db.execute("SELECT COUNT(*), SUM(tokens) FROM chunks").fetchone()
    if not total:
        raise SystemExit("chunks is empty -- run `python -m embed.chunk run` first")
    texts = [r[0] for r in db.execute(
        "SELECT text FROM chunks ORDER BY random() LIMIT ?", (n,))]
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  sample: {len(texts):,} of "
          f"{total:,} chunks  |  scope: {toks:,} tokens")
    for name in models:
        print(f"\n{name}")
        vecs = {}
        for dtype in dtypes:
            tps, dt, vram, vecs[dtype] = bench(name, texts, batch, dtype)
            print(f"    {dtype}: {tps:,.0f} tokens/s  ({dt:.1f}s for sample)  "
                  f"peak VRAM {vram:.1f} GB  ->  {toks / tps / 3600:.1f} h for this scope")
        if {"fp32", "fp16"} <= vecs.keys():
            cmin, ov = agreement(vecs["fp32"], vecs["fp16"])
            print(f"    fp16 vs fp32: min cosine {cmin:.5f}, "
                  f"top-10 neighbours kept {ov:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="embed.bench")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"],
                    choices=["fp32", "fp16"])
    a = ap.parse_args()
    main(a.n, a.models, a.batch, a.dtypes)
