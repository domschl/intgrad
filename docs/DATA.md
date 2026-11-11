# intgrad — Training Data (D4/D5)

> **Never `git add` anything under `data/`** — much of it is not openly
> licensed; research-only per Decision D5. The repo stores *pipeline code*,
> not data. `.gitignore` enforces this (`data/*`, allow `data/sample.txt`).

## What is on this machine (corpus v0)

| Source | Location | Size | Kind | License note |
|--------|----------|------|------|--------------|
| Wikipedia 2017 "definitions only" | `data/raw/wikipedia/wiki2017-defs.txt` | 2.7 GB | 5.3 M lead-paragraph definitions, `Title \|\|\| text` | en-wiki text, CC BY-SA; research use |
| Local text collection (en+de) | `data/raw/textcorpus/*.txt` | 31 MB | local en/de literature collection (not redistributed) | mixed; research use only |
| Project Gutenberg mirror (local) | `$CORPORA/Gutenberg` | 38 GB | full PG rsync (`aleph.gutenberg.org::gutenberg`) | public domain (US) mostly |
| PG mirror — curated copy in repo data dir | `data/raw/gutenberg-mirror/` | 37 GB | 89,010 books, 50 KB–15 MB filter | as above |
| WMT15 en⇄fr parallel | `$CORPORA/WMT15/` | ~4 GB | translation pairs (giga-fren + testsets) | research use; for later translation/multilinguality eval |
| enwiki/dewiki 2015 full-article dumps | `$CORPORA/Wikipedia/` | 12 GB bz2 (+54 GB extracted en) | full wiki articles, XML | CC BY-SA; needs an XML/markup extractor adapter (M4+) — far richer than defs-only |
| Teacher tool traces (generated) | `tools/gen_traces.py` → `data/traces.jsonl` | on demand | agent trajectories w/ rejection sampling | ours |

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
* **Protocol-safe payload (D2):** all C0 control bytes except `\t`/`\n`
  stripped (this includes every ESC-marker byte); `\r` normalized to `\n`.
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

## Next data moves (in order)

1. ~~Write `--wiki-xml` adapter~~ **DONE** — full enwiki+dewiki 2015 dumps
   → `data/corpus-v1-wiki/` (adapter: ns0 only, no redirects/disambigs,
   template/table/ref/link stripping, bz2 auto-detect, `LANG=PATH` specs,
   post-build manifest `verify` step must report `problems: 0`).
   Known v1 stripper artifacts: rare `[ ]` residues, template-glitch word
   joins — acceptable at scale; improve if val-bpb stalls.
   NOTE: corpus-v0 manifests were built with an off-by-SEP doc-offset bug
   (fixed later in the same session); v0 *content* is fine for random-byte
   training, rebuild `data/corpus-v0` before relying on its doc boundaries.
2. **Multilingual Latin corpus v1**: PG mirror already carries de/fr/es/it/
   pt/nl books (detector keeps them); optionally add FineWeb-2 / OSCAR
   slices via `uv run --with datasets …` when prose diversity beyond
   books is needed.
3. **Dialogue SFT**: Infinity-Instruct / OpenHermes / OpenAssistant-style
   sets via HF (`uv run --with datasets`), converted to the protocol
   byte-stream (D2) with loss masks on assistant turns.
4. **Tool SFT**: scale `tools/gen_traces.py` against the local teacher
   (D3), replay-filter, plus ToolACE/xLAM/ToolBench for variety.
5. **Val/test shards**: freeze held-out splits *before* any further
   augmentation; pin them by hash in the manifest.

## uv policy

All Python goes through `uv` (repo convention). Scripts carry PEP 723
inline metadata; run with `uv run tools/<script>.py`. Heavyweight HF
tooling: `uv run --with datasets,pyarrow …` (creates ephemeral envs,
nothing to install system-wide).
