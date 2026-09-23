# TanyaChaCham

Cited retrieval over Torah, Talmud and Chassidus. Answers link back to the
source text; the system never asserts Torah in its own voice.

## Why retrieval and not a trained model

The corpus is ~60-100M words -- about 2% of English Wikipedia, and roughly
1.2GB of embedding vectors. That is far too small to train a model that knows
anything, but ideal for retrieval. A model trained from scratch on Chassidus
would reproduce the *cadence* of Chassidus while fabricating attributions,
which is the one failure mode this corpus cannot tolerate.

## Design

**Sefaria refs are the universal address space.** Every segment and every
citation edge is keyed by a canonical ref (`Tanya, Part I; Likkutei Amarim 1:3`,
`Berakhot 2a:1`, `Genesis 1:1`). Sefaria has already hand-resolved the
intertextual citation graph between them, so we consume it rather than rebuild
it -- Tanya ch.1 alone carries 48 resolved citations.

**Language is first-class and never inferred from script.** Sefaria's
`language` field is really a *script bucket*: 42 Yiddish versions are filed
under "Hebrew", and German/French/Russian under "English". The per-version
`actualLanguage` field carries the truth. This matters because the Rebbe's
sichos were delivered in Yiddish and the published text largely preserves it.
Hence `segments` (an addressable location) is split from `texts` (one row per
ref x language x version).

## Layout

    store/schema.sql      works, segments, texts, links
    ingest/sefaria.py     per-work API ingest (superseded by bucket.py)
    ingest/bucket.py      parallel ingest from the GCS export -- the fast path
    ingest/links_bulk.py  5M-edge citation graph from links*.csv
    data/                 generated, gitignored (~3GB)

## Build the corpus

    python -m ingest.links_bulk fetch      # 700MB, 17 CSV shards
    python -m ingest.links_bulk load       # ~5.0M edges, ~15s
    python -m ingest.bucket run            # 19,754 versions, ~15-20 min

## Status

- [x] 5,043,902 citation edges loaded and indexed
- [x] Chasidut ingested: 457 versions, 0 failures, 23s
- [ ] Full ingest across all 19,754 versions
- [ ] Normalize complex-work node titles to base works
- [ ] Chunking + embeddings (GPU step)
- [ ] Retrieval + citation rendering
- [ ] Eval set: retrieval recall@k, citation accuracy

## Known gaps in the Sefaria layer

Chassidus on Sefaria is 40.3M Hebrew words against 7.1M English -- **5.6:1,
under 18% translated**. *Chanah Ariel*, *Sha'arei Avodah* and others are 0%.
*Etz Chaim*, the central Arizal text, is absent entirely. None of the Rebbe's
Yiddish material is here.

These gaps are what the chabad.org export is for. It is load-bearing, not
supplementary.
