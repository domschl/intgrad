# intgrad — Training Data

> **`data/` is private and is never committed** (plan/001 D5). Much of it is
> not openly licensed (research use only). The repo stores *pipeline code*,
> not data. `.gitignore` ignores `data/` and `runs/` with no exceptions.
> Repo-authored fixtures live in `tests/fixtures/`. Do not quote corpus text
> or model samples in docs, plans or commit messages.

All tools live in the uv project `py/` (`cd py && uv sync --extra corpus`).

## Raw sources on this machine

| Source | Location | Size | Kind |
|--------|----------|------|------|
| Wikipedia 2017 "definitions only" (`Title \|\|\| text`) | `data/raw/wikipedia/wiki2017-defs.txt` | 2.7 GB | 5.3 M lead paragraphs (en) |
| Local text collection (en+de) | `data/raw/textcorpus/*.txt` | 31 MB | literature |
| Project Gutenberg mirror | `data/raw/gutenberg-mirror/` | 37 GB | 89,010 files (50 KB–15 MB) |
| enwiki/dewiki 2015 full-article dumps | `$CORPORA/Wikipedia/` | 12 GB bz2 | XML |
| WMT15 en⇄fr | `$CORPORA/WMT15/` | ~4 GB | parallel text (unused) |

## Current data (plan/001 M1)

| Artifact | Contents | Size |
|---|---|---|
| `data/corpus-v2/` | cleaned books + wiki-2017 definitions + text collection | 21.5 GB, 5.27 M docs |
| `data/corpus-v1-wiki/` | en + de Wikipedia 2015 articles | 18.6 GB, 5.56 M docs |
| `data/split-v1/` | index v2 over both corpora, train/val/test buckets | 40.0 GB train |
| `data/eval-v1/` | frozen eval set: 512 × 2 KB chunks of val (and test) + baselines | 1 MB each |

`data/corpus-v0/` and `data/split-v0/` are superseded (v0 books contained
boilerplate and encoding-variant duplicates; split-v0 uses index v1, which
the readers no longer accept).

### corpus-v2 build

```sh
uv run --project py --extra corpus ig-build-shards --out data/corpus-v2 \
    --wiki-defs data/raw/wikipedia/wiki2017-defs.txt \
    --corpus-dir data/raw/textcorpus \
    --gutenberg-tree data/raw/gutenberg-mirror \
    --langs en,de,fr,es,it,pt,nl                 # ~6 min on 20 cores
```

Builder guarantees: control-free payload (all C0 except `\t`/`\n`
stripped, `\r` → `\n`; the range is reserved for in-band markers, plan/001
D7), encodings normalized to UTF-8, exact duplicates dropped, language gate
(function-word scoring on body windows), provenance per doc in
`manifest.jsonl` (`src`, `kind`, `lang`). Shards: docs joined by `\n \n`,
64 MB each.

### Project Gutenberg cleanup (`intgrad/corpus/gutenberg.py`)

PG files wrap the book in material that is not part of it. The cleaner
keeps everything that belongs to the book (title page, contents,
dedication, preface) and removes:

1. **License header/footer** by marker: modern `*** START/END OF … ***`
   (also wrapped over two lines), old `*END*THE SMALL PRINT!` headers,
   `End of (the) Project Gutenberg …` / `End of … E-text` lines. START is
   searched in the first min(64 KB, half) of the file, END only in the
   second half.
2. **Boilerplate paragraphs** in the head zone (first min(24 KB, 25%)) and
   tail zone (last min(32 KB, 25%)): production credits, PG/e-text
   mentions, proofreading teams, scan sources, URLs, HTML-version notes,
   license lines, errata lines (`page N … changed/corrected`); bracketed
   multi-paragraph transcriber notes as a unit; a trailing
   "Transcriber's Notes" section to the end of the file (only if it starts
   within the last 12 KB).
3. **Anywhere:** bracketed inline `[Transcriber's note: …]`, paragraphs
   that start with a transcriber note, and the World Library copyright
   block repeated between Shakespeare plays.
4. **Guard:** if steps 2–3 would remove > 30% of the marker-cut text, they
   are reverted for that book.

Measured on 3 × 400 random books (manual review of removed and residual
text): every reviewed removal was non-book material; residual PG-specific
text in 1–2 books per 400 (vs 95% before). Known edge case: a book *about*
digital libraries loses a few tail paragraphs.

**Variants and near-duplicates.** 27 k of 50 k numbered books exist as
`N.txt` / `N-8.txt` / `N-0.txt` (ASCII / 8-bit / UTF-8); one per id is kept
(UTF-8 > 8-bit > ASCII). Remaining duplicates (old `etextNN` names, other
editions) are found by content-defined shingle fingerprints over an
accent-folded letters-only view (≥ 50% of a book's sampled shingles owned
by one earlier book → dropped). Validation: 58/60 known variant pairs
detected, 0/300 distinct books flagged.

corpus-v2 build stats: 57,986 PG files after variant selection (31,024
dropped), 8,921 near-duplicates dropped, 46,737 books kept; 1.10 GB footer
+ 94 MB header + 28.5 MB paragraph boilerplate removed; guard fired on 30
books. Books shrank from ~35 GB (v0) to ~18.7 GB.

### split-v1 / index v2

```sh
uv run --project py ig-make-split --corpus data/corpus-v2 --corpus data/corpus-v1-wiki \
    --out data/split-v1 --wiki-defs-raw data/raw/wikipedia/wiki2017-defs.txt
# ... --check   rebuilds in memory and compares byte-for-byte
```

- Buckets: blake2b(key) % 1000, 1‰ val, 1‰ test. key = document bytes,
  **except English wiki (definitions and articles): the normalized article
  title**, so a 2017 definition and its 2015 article always share a bucket.
  Per-document hashing leaked: 30% of val articles had their lead in a train
  definition (B10). Keying on the lead text still leaked ~14% because of
  edits. With titles, overlap is < 1% and mostly templated stub articles.
- Index v2: shard paths relative to `idx.bin`, a source table (`books`,
  `wiki-defs`, `wiki-en`, `wiki-de`, `text`) and a source byte per doc;
  docs in corpus order. Readers: `src/dataset.c`, `intgrad.data.Index`.

| source | train bytes | val docs | test docs |
|---|---|---|---|
| books | 46.5% | 46 | 53 |
| wiki-en | 32.5% | ~3.8 k | ~4.0 k |
| wiki-de | 14.0% | ~1.7 k | ~1.7 k |
| wiki-defs | 6.9% | ~5.2 k | ~5.2 k |
| text | 0.1% | 1 | 0 |

(Val has few books, so per-source book numbers carry book-level variance.)

## Batchers (`intgrad/data.py`)

A bucket is read as a virtual stream: each document followed by **EOD =
0x03**.

- **RandomBatcher**: windows at uniform stream positions (every byte
  equally likely, windows may cross documents). Used for all plan/001
  training.
- **SequentialBatcher**: `B` streams, each reading its share
  `P_e[b::B]` of a seeded per-epoch document permutation consecutively;
  row b of batch t continues row b of batch t−1 byte-for-byte; a
  `doc_start` mask marks inputs that begin a document. Prepared for
  recurrent-state training (plan 002).
- Both deterministic and resumable (`state_dict()` is JSON).
- Throughput (B=64, T=1024, page cache): random 134 MB/s, sequential
  245 MB/s (`uv run python -m intgrad.data --index …`).

## Eval set eval-v1 (`intgrad/eval_set.py`)

```sh
uv run --project py ig-eval-freeze freeze    --index data/split-v1/idx.bin --out data/eval-v1
uv run --project py ig-eval-freeze baselines --index data/split-v1/idx.bin --eval data/eval-v1
```

512 chunks × 2048 bytes from val (seed 20261117) and test (seed + 1),
byte-uniform positions. Scored targets are positions 1..2047 of every
chunk, predicted from in-chunk bytes only. Results: bpb ± standard error
over chunks, plus per source. Hashes are pinned in plan/001 §7, and the
loader refuses a mismatching file.

## Known corpus artifacts

- v1 wiki stripper: rare `[ ]` residues and template-glitch word joins.
- Books: `[Illustration: …]` captions and ASCII tables are kept (they are
  part of the book). Front matter (title pages, contents) is kept on purpose.
- Wiki stubs from templates (villages, census places, species) are highly
  repetitive across articles. This isn't leakage, but it is easy text.
