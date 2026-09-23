-- TanyaChaCham corpus store.
--
-- Sefaria refs are the universal address space: every segment and every
-- citation edge is keyed by a canonical Sefaria ref string.
--
-- LANGUAGE MODEL: a segment is an addressable location; its TEXT exists in N
-- language versions. Sefaria's export has only two buckets ("Hebrew",
-- "English") which are really SCRIPT buckets -- 42 Yiddish versions are filed
-- as Hebrew, and German/French/Russian as English. The Rebbe's sichos were
-- delivered in Yiddish, so language must be first-class and must never be
-- inferred from script.

CREATE TABLE IF NOT EXISTS works (
    title       TEXT PRIMARY KEY,   -- "Tanya", "Berakhot", "Genesis"
    he_title    TEXT,
    category    TEXT,               -- "Chasidut", "Talmud", "Tanakh", ...
    cat_path    TEXT,               -- full category path, "/Chasidut/Chabad"
    base_work   TEXT,               -- complex works: node title -> base title
    is_complex  INTEGER DEFAULT 0
);

-- The addressable location. No text here.
CREATE TABLE IF NOT EXISTS segments (
    ref         TEXT PRIMARY KEY,   -- "Tanya, Part I; Likkutei Amarim 1:3"
    work        TEXT NOT NULL,
    category    TEXT,
    position    INTEGER             -- NOT reading order: set by whichever version
                                    -- landed first, and partial versions count
                                    -- from their own start. Use texts.position.
);

-- One row per (segment, language, version). A sicha may carry a Yiddish
-- original, a Hebrew rendering and an English translation simultaneously.
CREATE TABLE IF NOT EXISTS texts (
    ref           TEXT NOT NULL REFERENCES segments(ref),
    lang          TEXT NOT NULL,    -- ISO 639: he, yi, arc, en, de, fr, ru, ...
    script        TEXT,             -- hebrew | latin | cyrillic
    version_title TEXT NOT NULL,    -- "Kehot Publication Society", ... never "merged"
    source        TEXT,             -- 'sefaria' | 'chabad.org'
    body          TEXT NOT NULL,
    words         INTEGER,
    position      INTEGER,          -- ordinal within THIS version's file: the only
                                    -- reading order that holds for every version
    PRIMARY KEY (ref, lang, version_title)
);

-- Intertextual citation graph, bulk-loaded from
-- gs://sefaria-export/links/links*.csv (~5M edges).
-- NOTE: b_ref may be a RANGE ("Exodus 1:1-6:1"), resolved at query time.
CREATE TABLE IF NOT EXISTS links (
    a_ref       TEXT NOT NULL,
    b_ref       TEXT NOT NULL,
    a_work      TEXT,
    b_work      TEXT,
    a_category  TEXT,
    b_category  TEXT,
    link_type   TEXT,
    PRIMARY KEY (a_ref, b_ref)
);

-- Retrieval units, rebuilt wholesale by embed.chunk. A chunk is consecutive
-- segments of ONE version inside one section (chapter, daf, verse), so it
-- never spans two translations or two chapters. `spans` maps character ranges
-- of `text` back to segment refs: retrieval matches the chunk, but the answer
-- cites the exact segment. A segment too long for the model is split, and
-- every piece cites the whole segment.
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY,   -- row index into data/vectors/*.npy
    work          TEXT NOT NULL,
    lang          TEXT NOT NULL,
    version_title TEXT NOT NULL,
    first_ref     TEXT NOT NULL,
    last_ref      TEXT NOT NULL,
    spans         TEXT NOT NULL,         -- JSON [[ref, start, end], ...]
    text          TEXT NOT NULL,         -- normalized: no nikkud or ta'amim
    tokens        INTEGER
);

CREATE TABLE IF NOT EXISTS ingest_state (
    work        TEXT PRIMARY KEY,
    stage       TEXT,
    segments    INTEGER DEFAULT 0,
    updated_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_seg_work  ON segments(work);
CREATE INDEX IF NOT EXISTS idx_seg_cat   ON segments(category);
CREATE INDEX IF NOT EXISTS idx_txt_ref   ON texts(ref);
CREATE INDEX IF NOT EXISTS idx_txt_lang  ON texts(lang);
CREATE INDEX IF NOT EXISTS idx_link_a    ON links(a_ref);
CREATE INDEX IF NOT EXISTS idx_link_b    ON links(b_ref);
CREATE INDEX IF NOT EXISTS idx_link_aw   ON links(a_work);
CREATE INDEX IF NOT EXISTS idx_link_bw   ON links(b_work);
