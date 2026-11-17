# intgrad — Training Data

> **`data/` is private and is never committed** (plan/001 D5). Much of it is
> not openly licensed (research use only). The repo stores *pipeline code*,
> not data. `.gitignore` ignores `data/` and `runs/` with no exceptions.
> Repo-authored fixtures live in `tests/fixtures/`. Do not quote corpus text
> or model samples in docs, plans or commit messages.

## What is on this machine (corpus v0)

| Source | Location | Size | Kind | License note |
|--------|----------|------|------|--------------|
| Wikipedia 2017 "definitions only" | `data/raw/wikipedia/wiki2017-defs.txt` | 2.7 GB | 5.3 M lead-paragraph definitions, `Title \|\|\| text` | en-wiki text, CC BY-SA; research use |
| Local text collection (en+de) | `data/raw/textcorpus/*.txt` | 31 MB | local en/de literature collection (not redistributed) | mixed; research use only |
| Project Gutenberg mirror (local) | `$CORPORA/Gutenberg` | 38 GB | full PG rsync (`aleph.gutenberg.org::gutenberg`) | public domain (US) mostly |
| PG mirror — curated copy in repo data dir | `data/raw/gutenberg-mirror/` | 37 GB | 89,010 books, 50 KB–15 MB filter | as above |
| WMT15 en⇄fr parallel | `$CORPORA/WMT15/` | ~4 GB | translation pairs (giga-fren + testsets) | research use; for later translation/multilinguality eval |
| enwiki/dewiki 2015 full-article dumps | `$CORPORA/Wikipedia/` | 12 GB bz2 (+54 GB extracted en) | full wiki articles, XML | CC BY-SA; needs an XML/markup extractor adapter (M4+) — far richer than defs-only |

## corpus-v0 build (done)

```sh
uv run tools/build_shards.py \
    --out data/corpus-v0 \
    --wiki-defs data/raw/wikipedia/wiki2017-defs.txt \
    --corpus-dir data/raw/textcorpus \
    --gutenberg-tree data/raw/gutenberg-mirror \
    --langs en,de,fr,es,it,pt,nl
```

Builder: `tools/build_shards.py` (uv, PEP 723 → pulls lxml). Guarantees:
* **Control-free payload:** all C0 control bytes except `\t`/`\n`
  stripped; `\r` normalized to `\n`. The control range is reserved for
  in-band markers (plan/001 D7: `0x03` end-of-document, `0x00` padding).
* **Encoding normalized:** ascii/latin-1/cp1252 → UTF-8 (byte-level model
  must see consistent encodings).
* **PG boilerplate removed** (old and new header/footer variants).
* **Exact-duplicate docs dropped** (blake2b-128).
* **Language gate (D4):** function-word scoring on body windows
  (25/50/75% of doc — the PG credits header is English for every book
  language, so the head is deliberately not used).
* **Provenance (D5):** `manifest.jsonl` per shard: doc offsets, lengths,
  source paths, detected language. `dataset.json`: run stats.

Shard format: raw UTF-8 byte streams, docs joined by `\n \n`, 64 MB shards
(mmap-friendly, GPT-2 style shuffle-at-shard-granularity later).

## corpus-v1-wiki build (done)

```sh
uv run tools/build_shards.py --out data/corpus-v1-wiki \
    --wiki-xml de=$CORPORA/Wikipedia/dewiki-20151002-pages-articles.xml.bz2.bak \
    --wiki-xml en=$CORPORA/Wikipedia/enwiki-20151002-pages-articles.xml.bak \
    --min-doc-bytes 300
```

| Metric | Value |
|--------|-------|
| Articles kept | **5,562,134** (en 3,871,622 · de 1,690,512) |
| Size | 18.6 GB, 278 shards |
| Rejected | 1.02 M stubs (< 300 B), 767 exact dups; redirects/disambigs/ns!=0 filtered at parse |
| Manifest verify | 5.56 M docs checked, **0 problems** |
| Protocol-byte scan | 0 marker bytes in sampled shards |
| Runtime | ~25 min total (de 5.5 GB + en 13 GB) |

## Split and sampling

`tools/make_split.py` builds `data/split-v0/` (blake2b content buckets,
1‰ val / 1‰ test; `idx.bin` consumed by `src/dataset.c`). Train shares by
bytes: books ≈ 61%, wiki articles ≈ 33%, wiki definitions ≈ 6%. Documents
are sampled **proportional to their length** (every byte equally likely).
Uniform-per-document sampling, used before plan/001, over-weighted short
wiki definitions about 700× relative to books.

Planned (plan/001 M1): index v2 with relative paths and a per-doc source
id, a sequential (document-continuous) batcher for recurrent-state work,
and a frozen eval set (eval-v1) with baselines.

## Known corpus artifacts

- v1 wiki stripper: rare `[ ]` residues and template-glitch word joins.
- v0 books: occasional "Transcriber's Notes", catalogs, `[Illustration:`
  marks. Add quality filters if they show up in generations.
- corpus-v0 wiki-2017 definitions and corpus-v1 wiki-2015 articles cover
  the same topics; the split is by exact hash only. A spot check found no
  exact overlap, and a near-duplicate check is scheduled (plan/001 M1).

## uv policy

All Python goes through `uv` (repo convention). Scripts carry PEP 723
inline metadata; run with `uv run tools/<script>.py`. Heavyweight HF
tooling: `uv run --with datasets,pyarrow …` (creates ephemeral envs,
nothing to install system-wide).
