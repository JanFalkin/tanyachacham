# TanyaChaCham — working notes for Claude

Cited retrieval over Torah, Talmud and Chassidus. Every answer links back to a
source text. The system must never assert Torah in its own voice.

A fabricated or misattributed citation is the one failure this project cannot
absorb. An answer that carries the cadence of Chassidus and cites a source that
does not say it is worse than no answer at all. Every retrieval path is built so
that a citation can be checked back to the text it came from.

## Decisions already made — do not relitigate

**The end goal is the entirety of Jewish thought.** Sefaria (Phase 1) and
chabad.org (Phase 2) are the first source systems, not the destination — the
working corpus is Torah, Talmud and Chassidus *today*, but the north star is the
whole of Jewish thought. Read the phasing below as sequencing, not as the final
scope; "a Chassidus tool" or "a Sefaria wrapper" understates the ambition.

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
  Fixed: `ingest/bucket.py` writes one `works` row per node carrying
  `base_work`, so `COALESCE(base_work, title)` collapses a citation to its
  work. Node names come from `flatten()`'s own traversal, not from a second
  parse of `schema`, so they match the refs by construction. 100% of the 3,029
  distinct Chasidut link work-names resolve.
- **`CREATE TABLE IF NOT EXISTS` hides schema drift.** It is a silent no-op
  against a table that already exists, so a column added to `schema.sql` never
  reaches a live database — `works.base_work` was declared and missing for a
  week, and the first symptom was `no such column`. `python -m store.migrate`
  diffs the database against `schema.sql` (by executing it into `:memory:` and
  comparing `PRAGMA table_info`) and adds what is missing. `bucket.run` calls
  it first. Run it after editing `schema.sql`.
- **`/api/index` does not report `isComplex`.** Probe by trying the simple
  fetch and falling back to `/api/shape/<title>`.
- **Chassidic seforim are Hebrew, not Yiddish.** Tanya is loshon kodesh — 3
  Yiddish function-word hits in 21,433 words. They were *taught* orally in
  Yiddish but *written* in Hebrew. The sichos are the exception, so do not
  generalise either way from one work.
- **`actualLanguage` is authoritative but not always right.** The Tanya version
  `Español Tanya 32` is filed upstream as `actualLanguage: "en"`. The corpus
  has only 24 `es` rows, so Spanish sits inside the 80,948 `en` rows. Harmless
  today; at embedding time a Spanish chunk retrieved as English is a citation
  that looks right and reads wrong.

## Commands

There is no system `python`, only `python3` — use the venv, which is also where
the GPU packages will go:

    python3 -m venv .venv && source .venv/bin/activate

    python -m store.migrate               # reconcile DB with schema.sql
    python -m ingest.links_bulk fetch     # 700MB, 17 CSV shards, resumable
    python -m ingest.links_bulk load      # ~5.0M edges, ~15s
    python -m ingest.bucket plan          # what would be ingested
    python -m ingest.bucket run           # all 19,754 versions, ~15-20 min
    python -m ingest.bucket run --category Chasidut --workers 12

`data/` is gitignored and fully regenerable (~3GB). Never commit it.

## State

Done: 5,043,902 citation edges loaded and indexed; Chasidut ingested
(457 versions, 0 failures, 30s); language-versioned schema migrated;
`works` populated by the bucket path with `base_work` node rows, so work-name
normalization is done rather than pending.

Corpus holds **zero** `yi` rows across all 432,630 texts — empirical
confirmation that Sefaria carries no Yiddish material, not an inference from
version metadata.

Next: full 19,754-version ingest (Halakhah 5,272 / Talmud 4,526 / Mishnah
3,280 / Tanakh 3,176 are the bulk) → chunking and embeddings (GPU step) →
retrieval with citation rendering → eval set (retrieval recall@k, citation
accuracy).

## Why chabad.org is load-bearing

Sefaria's Chassidus is 40.3M Hebrew words against 7.1M English — **5.6:1, under
18% translated**. *Chanah Ariel* (754K words), *Sha'arei Avodah* (342K) and
others are 0%. *Etz Chaim*, the central Arizal text, is absent entirely. No
Yiddish material is present at all.

Three distinct gaps, not one: Yiddish sichos, English Chassidus, and
Arizal/Kabbalah. Phase 2 exists to close them.
