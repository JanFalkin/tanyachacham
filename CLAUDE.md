# TanyaChaCham — working notes for Claude

Cited retrieval over Torah, Talmud and Chassidus. Every answer links back to a
source text. The system must never assert Torah in its own voice.

A fabricated or misattributed citation is the one failure this project cannot
absorb. An answer that carries the cadence of Chassidus and cites a source that
does not say it is worse than no answer at all. Every retrieval path is built so
that a citation can be checked back to the text it came from.

## Decisions already made — do not relitigate

**Retrieval, not training.** The corpus is ~60–100M words (~2% of English
Wikipedia, ~1.2GB of vectors). Far too small to train knowledge into a model,
ideal for retrieval. A from-scratch model would reproduce the *cadence* of
Chassidus while inventing sources — the one failure this corpus cannot absorb.
`../train-llm-from-scratch` is a separate learning exercise; it is not this
project and we do not use it.

**Full corpus, not a Tanya pilot.** An earlier pilot-first plan was overruled:
Tanya is densely referential and needs the surrounding layer to be useful.

**Phase 1 = Sefaria. Phase 2 = chabad.org.** Split by *source system*, not by
book — that is where the real work boundary sits.

**Never infer language from script.** Sefaria's `language` field is a SCRIPT
bucket: 42 Yiddish versions are filed as "Hebrew", German/French/Russian as
"English". Use per-version `actualLanguage`. This is why `segments` (location)
is split from `texts` (one row per ref × language × version).

**Ingest individual versions, never `merged.json`.** Merging collapses versions
and keeps only the script bucket, so a "Hebrew" merged file can silently absorb
Yiddish with no way to separate it. It also destroys the Kehot-vs-Sefaria
translation distinction.

**Bucket, not API.** The GCS export has no rate limit and parallelises:
457 versions in 23s, versus hours of `/api/texts` calls. `ingest/sefaria.py` is
the superseded API path, kept for reference only.

## Gotchas that already cost time

- **Talmud refs are daf notation, not integers.** Sefaria indexes 1='1a',
  2='1b', 3='2a'; tractates start at 2a so indices 1–2 are empty. Emitting
  `Berakhot 3:1` instead of `Berakhot 2a:1` joins nothing.
- **Complex works with a blank node key** produced `Likutei Moharan,  272:1:1`
  (doubled space) instead of canonical `Likutei Moharan 272:1:1`. That was 8%
  of all segments matching zero links. Fixed in `flatten()`; keep it fixed.
- **Link `b_ref` may be a RANGE** (`Exodus 1:1-6:1`), not a single segment.
  Stored as-is; resolve at query time. Expanding at load loses the fact that
  the citation was to a span.
- **`links_by_book*.csv` are rollups**, not the edge list. Excluding them
  avoids double-counting.
- **Link work names ≠ our work titles for complex texts.** The CSV carries the
  node title (`Tanya, Part I; Likkutei Amarim`), never the base (`Tanya`).
  Joining on work identity needs a normalization pass — still TODO.
- **`/api/index` does not report `isComplex`.** Probe by trying the simple
  fetch and falling back to `/api/shape/<title>`.

## Commands

    python -m ingest.links_bulk fetch     # 700MB, 17 CSV shards, resumable
    python -m ingest.links_bulk load      # ~5.0M edges, ~15s
    python -m ingest.bucket plan          # what would be ingested
    python -m ingest.bucket run           # all 19,754 versions, ~15-20 min
    python -m ingest.bucket run --category Chasidut --workers 12

`data/` is gitignored and fully regenerable (~3GB). Never commit it.

## State

Done: 5,043,902 citation edges loaded and indexed; Chasidut ingested
(457 versions, 0 failures, 23s); language-versioned schema migrated.

Next: full 19,754-version ingest → work-name normalization → chunking and
embeddings (GPU step) → retrieval with citation rendering → eval set
(retrieval recall@k, citation accuracy).

## Why chabad.org is load-bearing

Sefaria's Chassidus is 40.3M Hebrew words against 7.1M English — **5.6:1, under
18% translated**. *Chanah Ariel* (754K words), *Sha'arei Avodah* (342K) and
others are 0%. *Etz Chaim*, the central Arizal text, is absent entirely. No
Yiddish material is present at all.

Three distinct gaps, not one: Yiddish sichos, English Chassidus, and
Arizal/Kabbalah. Phase 2 exists to close them.
