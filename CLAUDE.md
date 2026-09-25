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

**Retrieval, not training.** The corpus is 368M words across all versions (285M
Hebrew, 68M English) — ~8% of English Wikipedia; Chasidut plus what it cites is
361K chunks, ~1.5GB of vectors. Far too small to train knowledge into a model,
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

**The Divine Name is "G-d"** in everything the system writes in its own
voice, following Chabad convention. "Lord" stays as written. The *embedded* copy of every text
and every query is normalized the same way (`embed.chunk.normalize`): English
translations mostly write "God" (55K) against 5.6K "G-d" split over three dash
characters, so without it the question and the text disagree. The displayed
source text (`texts.body`) is never altered — a citation quotes the publisher.

**Footnotes are not the text.** Sefaria inlines a translator's footnote inside
the segment (`<sup class="footnote-marker">` + `<i class="footnote">`, which
may nest `<i>`). Stripping tags alone put the note mid-sentence in
`texts.body`, where a quotation presents the translator's gloss as the
source. `ingest.bucket.split_notes` moves them to `texts.notes` (JSON
`[[marker, note], ...]`); `body` and `words` are the text alone. It was 7.7M
of the 75.3M English words, across 249,642 rows.

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
- **`books.json` lists `merged.json` as if it were a version.** 8,097 of its
  19,754 entries have `versionTitle: "merged"`. Ingesting them doubled every
  word count (the first Chassidus figures were exactly 2× too high) and filed
  text under the script bucket. `load_books()` drops them; keep it that way.
- **`/api/index` does not report `isComplex`.** Probe by trying the simple
  fetch and falling back to `/api/shape/<title>`.
- **Chassidic seforim are Hebrew, not Yiddish.** Tanya is loshon kodesh — 3
  Yiddish function-word hits in 21,433 words. They were *taught* orally in
  Yiddish but *written* in Hebrew. The sichos are the exception, so do not
  generalise either way from one work.
- **The 4060 reports `power.draw` as `[N/A]`.** `gpu_stats()` parsed all
  three nvidia-smi fields at once, so one N/A made the temperature None and
  the GPU heat guard could never trip — silently. Fields now parse one by
  one, and `embed.embed` refuses to start without a GPU temperature.
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
    python -m ingest.bucket run           # all 11,657 versions, ~11 min
    python -m ingest.bucket run --category Chasidut --workers 12
    python -m embed.chunk run --scope cited  # Chasidut + cited sections, ~2 min
    python -m embed.bench                 # tokens/s per model on this GPU
    python -m embed.embed --model BAAI/bge-m3   # resumable, heat-guarded
    python -m eval.score                  # recall/MRR on eval/questions.jsonl
    python -m search.query "What are the seven Noahide laws?" [--category Chasidut] [--work Tanya] [--lang en]

PyTorch is the cu126 build: the GTX 1060 (sm_61) is gone from newer CUDA
builds. A Blackwell card (50-series) needs cu128+ instead.

`data/` is gitignored and fully regenerable (~3GB). Never commit it.

## State

Done: 5,043,902 citation edges loaded and indexed; full Sefaria corpus
ingested (11,657 versions, 6,462 works, 0 failures, ~9 min) — 5,054,293 text
rows over 3,486,015 segments, 284.6M Hebrew and 67.6M English words with
footnotes split out;
language-versioned schema migrated; `works` populated with `base_work` node
rows, so work-name normalization is done rather than pending.

Chasidut holds **zero** `yi` rows across its 216,446 texts. Sefaria's only
Yiddish is Tanakh (Yehoyesh's translation, 23,109 rows) plus 18 Mishnah rows —
so "no Yiddish" is true of Chassidus, not of Sefaria.

Chunked scope "cited": 360,514 chunks, 87.4M tokens (Hebrew ~2.0 tokens/word,
English ~1.5), none over 512. Talmud (incl. Rashi/Tosafot on it) is windowed
at 192 tokens, everything else at 384; an over-long segment is cut into equal
pieces, not a full piece plus a scrap. (Before footnotes were split out and
Talmud windows shrank: 322,011 chunks, 91.8M tokens.)

Embedded "cited" with both models on the 1060 (e5-base 3.2h, bge-m3 8.6h,
GPU max 68°C, no heat pauses). Scored on the 10 draft questions: bge-m3
recall@1 0.40 / @10 0.60 / MRR 0.50; e5-base 0.30 / 0.50 / 0.37. Too few
questions to choose a model. Misses traced to (1) chunks mixing topics
within a daf (Berakhot 61b:2 sits after the organs passage, rank 213),
(2) inline footnotes in Sefaria English, (3) dash variants of "G-d" — the
"G-d" normalization is committed but the stored vectors predate it.

RTX 4060 8GB installed (sm_89, driver 580, same cu126 build, PCIe 4.0 x4).
`embed.bench` on the "cited" scope, 2,000-chunk sample:

    model      fp32 tok/s  fp16 tok/s  fp16 hours  fp16 vs fp32
    e5-base        30,375     102,519       0.2    min cos 0.9995, top-10 kept 97.9%
    bge-m3          9,268      32,571       0.8    min cos 0.9997, top-10 kept 99.5%

Re-embedded the new chunks in FP16 on the 4060 (e5-base ~15 min, bge-m3
~45 min, GPU max 61°C, no heat pauses). Old vectors kept in
`data/vectors-1060-2026-09-23/` (they pair with the old chunks, so they can
no longer be scored). On the 10 draft questions:

    model     before (1060, old chunks)   now (4060 fp16, new chunks)
              R@1   R@10  MRR             R@1   R@5   R@10  MRR
    bge-m3    0.40  0.60  0.50            0.20  0.70  0.70  0.46
    e5-base   0.30  0.50  0.37            0.50  0.50  0.60  0.53

Berakhot 61b:2 went from rank 213 to 2 under bge-m3 (8 under e5): the Talmud
windows did what they were for. bge-m3 now has five questions at rank 2 --
beaten by a neighbouring source on the same idea (Zohar on the three
garments, Eruvin 54a on Deut 30:14, Flames of Faith on two souls). Whether
those count as hits is a judgment for the eval review, and 10 questions still
cannot choose a model. Weak everywhere: t006 (joy; only Tanya counted, >100)
and t010 (the Hebrew "beinoni" question; bge-m3 27, e5 >100).

Next: fold in Meir's review of eval/questions.jsonl, and widen the eval set
well beyond 10 questions -- the next change to chunking or model should be
chosen by it, not by these ten.

## Why chabad.org is load-bearing

Sefaria's Chassidus is 20.4M Hebrew words against 3.5M English — **5.8:1, under
18% translated**. *Chanah Ariel* (377K words), *Sha'arei Avodah* (171K) and
others are 0%. *Sefer Etz Chaim*, the central Arizal text, is filed under
Kabbalah: 344K Hebrew words, 30K English (~9%). No Yiddish Chassidus is present
at all.

Three distinct gaps, not one: Yiddish sichos, English Chassidus, and
Arizal/Kabbalah. Phase 2 exists to close them.
