# TanyaChaCham

A source-finder for Torah, Talmud and Chassidus. Ask a question, in English
or Hebrew, and it returns the passages that address it -- each quoted exactly
as its publisher printed it, under its exact reference, so every result can
be checked against the page it came from.

It is a study aid, not a posek. It does not answer in its own voice, and it
never presents a teaching without the source that says it.

## Principles

- **Every result is a citation.** A passage is shown under the exact
  reference it came from (`Tanya, Part I; Likkutei Amarim 12:1`,
  `Sanhedrin 56a:24`), in the named translation it came from.
- **Quoted text is never altered.** What is displayed is the publisher's
  wording. Search uses a separate, normalized copy (for example "God" and
  "G–d" both become "G-d", so a question and a text agree); that copy is
  never shown.
- **A translator's footnote is not the text.** Footnotes are stored apart from
  the passage they annotate, so a note is never quoted as the author's words.
- **Translations are never merged.** Each edition and translation is kept
  separately; one passage in four translations is one result, with each
  translation named.
- **Language is recorded, not guessed.** Yiddish and Hebrew share a script, so
  the script says nothing about the language. Each text carries the language
  it was actually written in.
- **No source, no answer.** A fabricated or misattributed citation is worse
  than no answer at all. The design choices above exist so that one never
  happens unnoticed.

## Searching

    python -m search.query "What are the seven Noahide laws?"
    python -m search.query "What are the three garments of the soul?" --category Chasidut --work Tanya
    python -m search.query "..." --lang en --k 10

Each result shows the passage, its reference and translation, and the other
translations of the same passage:

     4. Sanhedrin 56a:24   score 0.581
        [en] William Davidson Edition - English
        The Sages taught in a baraita : The descendants of Noah, i.e., all of
        humanity, were commanded to observe seven mitzvot: ...
          also [en] Sefaria Community Translation  (0.554)
          also [de] Talmud Bavli. German trans. by Lazarus Goldschmidt  (0.549)

`--work` and `--category` narrow the search before ranking. They matter for
attribution as well as convenience: "the Rebbe" in *Sichot HaRan* is Rabbi
Nachman of Breslov, and similarity search matches the word, not the person.

## What is in it today

The complete public Sefaria library: 11,657 text versions of 6,462 works,
5,054,293 passages (284.6M Hebrew and 67.6M English words), and Sefaria's 5.0M
resolved citations between them -- Tanya chapter 1 alone carries 48.

Search currently covers Chassidus and every passage it cites: 360,514 search
passages. Retrieval is measured against a small draft set of questions whose
correct sources are known; the set is under rabbinic review and will grow.

## What is missing

Sefaria's Chassidus is 20.2M Hebrew words against 3.3M English -- about 6:1,
16% translated. *Chanah Ariel* (377K words) and *Sha'arei Avodah* (167K) have
no English at all; *Etz Chaim*, the central Arizal text, is 9% translated.
Sefaria holds no Yiddish Chassidus, and none of the Lubavitcher Rebbe's works:
no Likkutei Sichos, Toras Menachem, Sefer HaMaamarim, Igros Kodesh or Hayom
Yom. A question about what the Rebbe taught cannot be answered from it.

Closing those gaps -- the Rebbe's teachings, Chassidus in English, and the
Yiddish originals -- is the next phase. The destination beyond that is the
whole of Jewish thought; Sefaria is the first source, not the last.

## Texts are not published here

This repository holds code only. The texts, citations and search vectors are
built locally into `data/`, which is excluded from git and never published.
Quoted text is shown to a reader with its source and a link to it.

## How it is built

Retrieval, not a trained model: a model trained on Chassidus would learn its
cadence while inventing its sources. Instead, passages are embedded as vectors
by a multilingual model (bge-m3 or multilingual-e5), a question is embedded
the same way, and the closest passages are returned with their references.

    store/schema.sql      works, segments, texts (+ footnotes), links, chunks
    store/migrate.py      reconcile a live database with schema.sql
    ingest/bucket.py      the Sefaria library, one version at a time
    ingest/links_bulk.py  the 5M-edge citation graph
    embed/chunk.py        passages -> search chunks that cite exact segments
    embed/embed.py        embed chunks on the GPU; resumable, heat-guarded
    embed/bench.py        embedding speed and fp16/fp32 agreement
    eval/questions.jsonl  questions with the sources that answer them
    eval/score.py         recall@k and MRR against those questions
    search/query.py       the command-line search

Build from scratch (Python 3, CUDA GPU; see `CLAUDE.md` for detail):

    python3 -m venv .venv && source .venv/bin/activate
    python -m ingest.links_bulk fetch && python -m ingest.links_bulk load
    python -m ingest.bucket run                         # ~9 min
    python -m embed.chunk run --scope cited             # ~2 min
    python -m embed.embed --model BAAI/bge-m3 --fp16    # ~45 min on an RTX 4060
