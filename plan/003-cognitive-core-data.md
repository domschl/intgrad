# Plan 003 — Data for a cognitive core

*Drafted 2026-11-21 after plan 002 N3c. Status: **active** (draft
decisions settled the same day, 003-D5 to D9). Plan 001 stays the
reference for numerics, eval rules and experiment rules (§4.5 there), and
plan 002 for the model (N2b, cosine, Lion). Results go into §8 of this
plan.*

## 0. Goal and scope

The target is a **cognitive core**, not a knowledge store. The aim is the
essentials:
- fluent use of at least one language, up to paragraph-level coherence;
- reading and writing structured formats (code, JSON/XML tool calls);
- some reasoning.

Facts the model can look up through tools are not the goal.

The model side has stopped being the bottleneck. Plan 002 N3c measured:
- 3× the data gave −0.083 on eval-v1;
- 12 blocks gave a further −0.018;
- every architecture variant was within ±0.01 or worse.

So this phase spends its effort on **what the model reads**. It answers
three questions:
1. **Language:** one dominant language, or several that amplify each
   other into a more abstract core?
2. **Structure:** does code and structured data (JSON, XML, tool calls)
   help language and reasoning at this scale, as reported for larger
   models?
3. **Quality:** do textbook-like books beat the current Gutenberg- and
   Wikipedia-heavy pool at equal bytes?

It ends with the **best-case fp model** that fits the 12 GB card (C5).
That is the reference plan 002 N4 (byte numerics) has to match.

**Out of scope:**
- post-training (instruction and tool-use tuning, reasoning traces);
- tokenizers and word-level anything.

Tool-use formats are *seen* in pretraining, but the model is not taught to
act on them here.

**Standing rules:**
- **Privacy.** data/ is private: it holds sources under proprietary
  licenses. It is never committed, and no source text is quoted in the
  repo, in plans or in commit messages. Plans give folder-level counts and
  sizes only, never titles. Generated samples stay local (runs/).
- **Language-agnostic modelling.** Bytes in, bytes out, no word
  boundaries. Language tags exist only as *data-side metadata*, for
  mixing and per-language eval. The model never sees them.

## 1. Starting point

**Current pool** (split-v1 train, 40 GB):

| source | bytes | share |
|---|---|---|
| books (Project Gutenberg, mostly pre-1930 English prose) | 18.6 GB | 46% |
| wiki-en | 13.0 GB | 32% |
| wiki-de | 5.6 GB | 14% |
| wiki-defs | 2.8 GB | 7% |
| text | 0.03 GB | – |

What this means for a cognitive core:
- German exists only as encyclopedic text. That explains why German
  samples lag English (user observation, N3c L12).
- There is no code, no structured data and no modern explanatory prose.

**The batcher mixes by pool composition.** `SequentialBatcher` streams
whole documents from a permutation of the bucket. A source's share of
training is its share of bytes, and there is no weight per source. C1 adds
weights.

**Model reference** (plan 002 N3c, seed 42, cosine):

| model | data | eval-v1 |
|---|---|---|
| N2b, 36 M | 300 MB | 1.490 |
| N2b, 36 M | 900 MB | 1.407 |
| N2b-L12, 54 M | 900 MB | 1.389 |

## 2. Sources

Every source gets the same treatment:
- a `kind` and `lang` per document in the manifest;
- a **license class**: open, permissive-code, copyleft, proprietary or
  unknown.

The license class is recorded per document. Every class stays in data/,
under the privacy rule.

### 2.1 Local: Markdown books (proprietary, private)

- Location: `~/Docs/KnrsData/MarkdownBooks`.
- Size: 3,488 books in 38 topic folders, 2.6 GB.
  - Median book: 450 kB. Six books are over 20 MB (largest 104 MB).
  - The largest folders are a general non-fiction folder (611 MB),
    history (351 MB), religion (215 MB), philosophy (201 MB), physics
    (140 MB), programming (129 MB), mathematics (76 MB) and AI (53 MB).
  - Fiction and literature, English and German, add about 440 MB in
    total.
- Language (300-book sample, function-word count): about 88% English and
  11% German. The German books are a *different register* from wiki-de:
  literature, psychology, cooking, advice.
- Markup (same sample):
  - tables 6% of bytes;
  - headings 0.7%;
  - about 28 code fences per book;
  - LaTeX in maths and physics;
  - almost no images, no inline data URIs;
  - some U+FFFD replacement characters (≈ 2 per MB).
- Cleaning (new kind `mdbook`):
  - keep Markdown (structure is the point) but normalise it;
  - collapse runs of table padding;
  - drop image links, front-matter boilerplate (copyright pages, tables of
    contents) and indices;
  - strip U+FFFD;
  - split very large files into chapter-sized documents.
- Overlap: the fiction and epic folders may duplicate Project Gutenberg
  books, so they get a near-duplicate check against corpus-v2.
- Value: the closest thing to "textbook quality" available. Explanatory,
  modern, mostly English, with German narrative prose.

**First score (`ig-mdbook-score`, 2026-11-21).** Classes per D6. The
per-book list stays in `data/mdbooks/scores.jsonl`.
- About 40% of the books are hard-wrapped (paragraphs split into short
  lines). The score works on blank-line separated blocks, and the cleaner
  unwraps them.
- Code fences and display maths count as content, not as damage.

| class | books | bytes |
|---|---|---|
| keep | 2,840 | 2.32 GB |
| reference-like | 403 | 0.28 GB |
| poor conversion | 245 | 0.15 GB |

Kept bytes by topic group and language (MB):

| group | en | de |
|---|---|---|
| stem | 370 | 29 |
| humanities | 451 | 114 |
| narrative | 293 | 134 |
| religion | 324 | 10 |
| practical | 3 | 2 |
| general | 534 | 26 |

The thresholds are first guesses. They get checked by user spot checks
on the per-book list. Group weights are set in C2: religion and the
general group are candidates for down-weighting, as knowledge-dense.

### 2.2 Local source repositories: not used (003-D7)

`~/Source` (263 git repositories) was surveyed, then excluded. Even
files tracked by git can hold secrets, personal notes or private data,
and checking them by hand is not practical. Code comes from public,
permissively licensed datasets only (§2.3).

### 2.3 Public datasets (downloaded into data/raw)

These are candidates. Each one's license is checked at download, and the
exact dataset, version and license go into the manifest.

| need | candidates | role |
|---|---|---|
| code, permissive | Stack v2 / StarCoder-data style permissive subsets (Python, C/C++, JS/TS, Rust, Go, shell, notebooks) | pretraining, 1–3 GB |
| tool use, JSON | xLAM function-calling 60k; Glaive function-calling v2 | converted to the D9 format; format exposure in pretraining, most kept for post-training |
| tool use, XML-tagged | Hermes function-calling v1 (`<tool_call>` style) | same |
| explanatory English | FineWeb-Edu sample; Cosmopedia (synthetic textbooks) | quality comparison against the Markdown books |
| German, modern | FineWeb-2 German subset | German register beyond Wikipedia |
| maths | FineMath / OpenWebMath sample | reasoning-dense text |
| simple narrative | TinyStories | coherence probe and ablation only |

**First download (C0, `ig-fetch`, 2026-11-21): 19.9 GB of parquet/JSON,**
pinned by revision with license and sha256 in `data/raw/hf/SOURCES.jsonl`,
archived to `~/Storage/Library/Corpora/HuggingFace`. Gated sets use the
user's `hf` login (token never logged, dropped on CDN redirects) once their
terms are accepted: xLAM (tool calls) and StarCoderData (about 7 GB:
Python, C, C++, JS, TS, Rust, Go, shell, JSON, YAML, Jupyter; permissive
licenses only, deduplicated, PII-redacted). StarCoderData is the preferred
code source, and github-code-clean is the ungated fallback.

| name | dataset (files) | license | raw |
|---|---|---|---|
| edu-en | FineWeb-Edu sample-10BT (2 of 14) | ODC-By | 4.3 GB |
| cosmo | Cosmopedia khanacademy, openstax, stanford, wikihow | Apache-2.0 | 4.2 GB |
| cosmo-math | Cosmopedia auto_math_text (2 of 18) | Apache-2.0 | ≈ 0.5 GB |
| cosmo-stories | Cosmopedia stories (2 of 43) | Apache-2.0 | ≈ 0.55 GB |
| web-de | FineWeb-2 deu_Latn (1 of 181) | ODC-By | 4.8 GB |
| math | FineMath-4plus (8 of 64) | ODC-By | 2.3 GB |
| code | github-code-clean (8 of 880; filtered to permissive licenses at build) | per file | 2.9 GB |
| tool-glaive | Glaive function-calling v2 | Apache-2.0 | 0.27 GB |
| tool-hermes | Hermes function-calling v1 | Apache-2.0 | 0.07 GB |

Tool-use data is conversational. Pretraining sees it as plain documents in
its own format (JSON or tagged), with no role semantics yet. Role and
turn conventions are a post-training decision (D5). The C0 control range
is already free for in-band markers (plan 001 D7), so no format choice is
locked in.

## 3. Pipeline changes

- **C0a, ingestion:**
  - `ig-build-shards` gains three kinds:
    - `mdbook` (§2.1, with the D6 suitability score);
    - `code` (§2.3 public code: one document per file, the path as a
      header line, dedup and the generated-file filters);
    - `jsonl` / `parquet` (§2.3: a text field and metadata per record;
      pyarrow as an optional extra; tool-use records converted to D9).
  - Every kind keeps the existing guarantees: no C0 bytes except `\t` and
    `\n`, exact dedup, provenance per document.
  - **Status (2026-11-21): implemented** in `intgrad.corpus.ingest`, used
    by `ig-build-shards --mdbooks … --hf NAME … [--exclude-gated]`; output
    `data/corpus-v3`. Per document: `kind`, `lang`, `lic`, `gated`,
    `private`, and `grp` (the split group key). Code also gets `repo` and
    `path`.
    - Kinds: mdbook, edu, cosmo, web, math, code, data (JSON and YAML
      files, weighted apart from code), tool.
    - Tool rendering for pretraining: role blocks `<system>` `<user>`
      `<assistant>` `<tool_response>`, schemas in `<tools>`, calls as
      `<tool_call>` + JSON (D9). Glaive's string-encoded arguments are
      parsed into JSON; 130 of 112,960 Glaive conversations do not parse
      and are dropped. xLAM and Hermes render completely.
    - Code filters: size, longest line (minified), alphanumeric share,
      generated-file markers, lock files, an 8 kB cap for JSON/YAML. They
      keep 95–100% of Python, shell and YAML, 61–66% of JSON and Jupyter.
      github-code-clean (60% kept, permissive only, e-mails and IPv4
      redacted) is left out of the default build; StarCoderData covers
      code.
    - Book cleaning: on a 60-book sample, hard-wrapped blocks drop from
      18% to 2% of blocks; the median size change is −0.2%.
    - Not done yet: near-duplicate check of the books against Project
      Gutenberg, front-matter stripping.
- **C0b, split-v2:** group keys for the split.
  - book: normalised title;
  - code: repository (public datasets carry the repository name);
  - wiki: title, as today;
  - tool-use: conversation id.
  - split-v1 and eval-v1 stay frozen for continuity.
- **C0c, eval-v2:** a frozen eval set with one source per domain:

  | source | contents |
  |---|---|
  | books-pg | Project Gutenberg books |
  | books-md | Markdown books, split into en and de |
  | wiki-en | as in eval-v1 |
  | wiki-de | as in eval-v1 |
  | code | per language group |
  | tool | JSON and XML tool calls |
  | edu-en | explanatory English (§2.3) |
  | web-de | modern German (§2.3) |

  All scored at 2048-byte chunks, as eval-v1. eval-v1 is still scored at
  the end of every run, so comparisons with plans 001 and 002 stay
  possible.
- **C1, mixture weights:**
  - `SequentialBatcher` takes a weight per source: streams are assigned to
    sources in proportion to the weights, and each stream reads whole
    documents from its source's permutation;
  - resume stays exact.
  - The weights are recorded in the run config and shown in ig-dash.
- **Probes** (deterministic, minutes on the GPU):
  - P5, brackets and indentation: scored on held-out code;
  - P6, format validity: the share of sampled JSON tool calls that parse;
  - coherence: a fixed-prompt sample set per language, reviewed by the
    user, local only.

## 4. Milestones

### C0 — Sources, pipeline, eval-v2
- C0a ingestion and C0b split-v2 (§3).
- The downloads in §2.3, each license recorded.
- eval-v2 frozen.
- Baselines: score the N3c checkpoints (N2b-900m, L12-900m) on eval-v2.
  This needs no training and shows where the current model is weakest.

### C1 — Mixture control
- **Status (2026-11-21): implemented.** `data.weights = {source = w}`
  with the sequential batcher.
  - Streams are assigned to sources by largest remainder, with at least
    one stream per weighted source, interleaved across the pool.
  - Each stream reads whole documents from its source's own per-epoch
    permutation.
  - Byte shares therefore follow the weights to 1/S. Tested at 50/50 and
    25/75 within ±1%.
  - Without weights the batcher is unchanged, bit for bit.
  - Resume is exact; a checkpoint with other weights is refused.
- Weighted batcher (§3), tests for exact resume and for the realised
  weights.

### C2 — Mixture ablations (36 M, 300 MB, seed 42, about 1.4 h each)
Configs `configs/c2-m1.toml` … `c2-m4.toml` (N2b protocol on split-v2,
eval-v2), runner `runs/c2.sh`. Byte shares (%), as realised by the
batcher:

| source | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| books-md-en | 15 | 12 | 11.4 | 17.7 |
| edu-en | 20 | 16 | 15.2 | 23.5 |
| cosmo | 15 | 12 | 11.4 | 17.7 |
| books (Gutenberg) | 15 | 12 | 11.4 | 17.7 |
| wiki-en | 10 | 8 | 7.6 | 11.7 |
| math | 10 | 8 | 7.6 | 11.7 |
| books-md-de / web-de / wiki-de | 5 / 5 / 5 | 4 / 4 / 4 | 3.8 / 3.8 / 3.8 | – |
| code | – | 20 | 18.9 | – |
| data / tool | – | – | 2.4 / 2.4 | – |

M0 is N2b s42 (old pool), rescored on eval-v2 (§8 C0).

Caveat of the sequential protocol: each of the 1024 streams reads about
290 kB in a 300 MB run. Long sources are therefore sampled from their
beginnings: about 150 Markdown books and 150 Gutenberg books per run at
15%. This holds equally for every mixture, but the per-source numbers for
books measure few books.

Compared on eval-v2 per source and on the probes:

| run | mixture (byte shares, set in C2) |
|---|---|
| M0 | current pool (plan 002 N2b, rescored on eval-v2) |
| M1 | English-dominant quality mix: Markdown books, edu-en, Project Gutenberg, wiki-en; German about 15% (books-md-de, web-de, wiki-de) |
| M2 | M1 + about 20% code |
| M3 | M2 + about 5% structured and tool data |
| M4 | M1 without German: does German help or dilute English? |

The winner is set by English and code bpb and by the probes. German and
tool-use bpb come second. Close calls get a second seed.

### C3 — Language question (only if M1 vs M4 is unclear)
- English-only vs English + 15% German vs English + 30% German, scored on
  the *English* sources.
- A small model is expected to dilute; transfer would show as better
  English bpb with German in.

### C4 — Equal-time control (from plan 002 N3c)
- 36 M on 1.35 GB vs 54 M on 900 MB (same 6.4 h), on the C2 winner.
- It decides whether the best-case run spends its hours on data or on
  model size.

### C5 — Best-case fp model
- The largest model the card trains, on the C2 winner.
- The budget is set by the user. For example, 54 M on 2.7 GB is about 19 h;
  the larger sizes are in plan 002 N3c's table.
- Its numbers (eval-v1, eval-v2, probes, samples) are the fp reference for
  plan 002 N4.

### C6 — Warm-started add-ons; revisit the nonlinear recurrence (after C2)
Agreed 2026-11-21. New parts are added to a trained model **without
changing its function**: the new branch's output projection into the
residual stream starts at zero, so at step 0 the grown model computes
exactly what the base model computed. Gradients reach the zero projection
at once; with Lion it moves by one learning-rate step on the first
update, so the branch does not stay stuck.

Base: the C2 winner (N2b on the winning mixture). Protocol for every
warm start:
- **Control:** the base model trained on for the same steps, on the same
  data. The grown model has already seen the base's data, so it is
  compared with continued training, never with a model trained from
  scratch.
- **Schedule:** a short warmup and a cosine decay over the added steps.
  A lower learning rate for the inherited weights is checked once, so
  Lion does not knock them off their optimum.
- **Scored on** eval-v2 per source, the probes (P5, P6) and ig-position.

Code needed: an option to start `RecMix.out` (and other new branches)
at zero; loading a checkpoint into a grown model (matching weights
copied, new ones fresh); a check that the grown model's outputs equal
the base's at step 0.

Runs:
1. **C6a, recurrence on a flat model.** Flat base (the C2 mixture without
   `rec`), recurrence inserted at zero in every block, carried-state
   protocol. Compared with the continued flat control and with N2b
   trained with recurrence from the start. This tells us whether a larger
   final model can be trained flat first and get its recurrence
   afterwards.
2. **C6b, nonlinear recurrence (plan 002 N3b).** Shelved on the old pool
   at 36 M. Code and structured data are where nonlinear state tracking
   should show (nesting, indentation, names defined far away). The layer
   runs at 62% of N2b's speed, so it is tested as a warm start:
   - base: the C2 winner;
   - the middle block's recurrence becomes the nonlinear one, with W and
     the new output path at zero, so it starts as the trained linear
     layer;
   - control: the same insertion with W but no tanh;
   - scored on eval-v2 code, data and tool sources and on P5.

   Only if this gains does the block-diagonal kernel work from plan 002
   come back.

### C7 — Patch memory: attention over a longer history (after C6a)
Agreed 2026-11-21, as the reusable form of the hourglass idea. Copying
flat weights into a patch-level core is not a reuse (its inputs and
positions change), so plan 001 M6's hourglass stays shelved.

Instead, the flat model stays whole as the byte-level part (1024-byte
windows). A small patch-level model is added beside it:
- it reads a longer history, the 4–8 kB before the window, in patches of
  k bytes;
- its output enters a middle block through a zero-initialised projection
  (cross-attention or an additive summary), so step 0 is the flat model.

It is a second cross-window memory: a full attention view of the recent
past, where the recurrence keeps a compressed state. Compared on the same
base and warm-start protocol as C6:

| run | cross-window memory |
|---|---|
| flat, continued | none |
| C6a | recurrence |
| C7 | patch memory |
| C7 + recurrence | both |

Scored on eval-v2, the carried and cold windows of ig-position, and the
cost (speed, memory). New code (the patch model and its injection); it
runs only after C6a has validated the warm-start protocol.

## 5. Decisions (plan 003)

- **003-D1 Order:** data first, then the best-case fp model, then plan 002
  N4 (user, 2026-11-21).
- **003-D2 Privacy:** every new source lives in data/ under the existing
  privacy rule, whatever its license; the license class is recorded per
  document. No titles or text in the repo.
- **003-D3 Splits:** grouped (book, repository, conversation), so no work
  straddles train and eval.
- **003-D4 Primary eval (revised 2026-11-21, user):** eval-v2 is the eval
  of every run on split-v2: periodic, quick validation and final.
  - Runs are compared per source; the equal-weight overall number is only
    a summary.
  - eval-v1 is not scored in training. It is byte-proportional to the old
    pool (about 50% Gutenberg, 45% Wikipedia), so it would penalise a
    mixture shift by design.
  - Continuity: eval-v2 contains eval-v1's four sources, and older
    checkpoints are rescored on eval-v2 with `ig-score` (§8 C0).
  - eval-v1 stays frozen on disk.
- **003-D10 Acknowledgment and open rebuild** (user, 2026-11-21):
  - every source, its license, the obligations we follow and the
    citation its authors ask for are in `DATASETS.md`, generated by
    `ig-acknowledge` from the fetch registry and the dataset cards;
  - `ig-fetch --open-only` re-downloads only the ungated sources, and
    `SOURCES.jsonl` marks gated files. Corpus builds take a source filter,
    so a corpus without the gated and private sources can be regenerated;
  - code ingestion keeps repository, path and license per document (The
    Stack terms), and the local StarCoderData copy follows the
    maintainers' removal updates.

- **003-D5 Language:** English-dominant, German a minority (about 15%,
  measured in C2 M4 and C3), because of capacity at this size (user,
  2026-11-21).
- **003-D6 Markdown books, selected by suitability** (user, 2026-11-21).
  There is enough text, so filtering is fine. Per book, not per folder:
  - kept: explanatory and narrative prose (science, maths, programming,
    philosophy, psychology, language, fiction and literature, manuals);
  - dropped or down-weighted: reference-like and knowledge-dense books
    (catalogues, chronologies, glossaries, liturgical and transliterated
    texts), and books whose conversion is poor (table share, broken
    characters, prose share below a threshold);
  - the per-book list with its scores lives in data/ only; the plan
    records counts and bytes per class.
- **003-D7 No local code:** `~/Source` is not used (secrets or private
  data even in tracked files); code comes from public permissive
  datasets (user, 2026-11-21).
- **003-D8 Download budget:** about 30 GB raw under data/raw, about 10 GB
  cleaned (user, 2026-11-21).
- **003-D9 Tool-call format:** an XML-style tag around a JSON body, e.g.
  `<tool_call>{"name": …, "arguments": {…}}</tool_call>`, with the
  result in `<tool_response>…</tool_response>` (the Hermes/Qwen
  convention). Pure-JSON-only samples are converted to it; JSON-only
  pretraining data (schemas, API payloads) stays as it is.
  Reasons (decided on request, 2026-11-21):
  - small models fail most at the *edges* of a structured span: knowing
    when a call starts and when it is complete. A literal tag is a fixed
    byte string, so the start and end are trivial to learn and to detect
    in sampling, with no depth counting;
  - inside the call, JSON is the standard for tool schemas and arguments,
    so the model sees the same syntax in pretraining code and data;
    arguments in XML would add verbosity and a second escaping scheme;
  - what made JSON hard for older models was mostly escaping (quotes and
    newlines inside strings) and closing nested braces. Both are short
    range inside a call, and P6 (§3) measures them directly;
  - sampling can later be constrained to valid JSON inside the tags
    without touching the model.
  Role and turn markers stay a post-training decision.

## 6. Open decisions

*(none: D1–D5 of the draft settled 2026-11-21, see 003-D5 to D9)*

## 7. Risks

- **License mix.** Proprietary books can be used
  privately but never distributed. Generated samples can echo them, so
  they stay local too. The license class per document makes any later
  public model build filterable.
- **Eval drift.** A new mixture moves bpb on every source. Comparisons go
  through eval-v2 per source, never one pooled number, and eval-v1 is kept
  for continuity.
- **Code dominance.** Code is low-entropy and pulls bpb down. A mixture
  is never judged on pooled bpb, only per source and on the probes.
- **Public code quality.** Permissive-code datasets still contain
  generated files, minified JS and vendored copies. The same filters
  apply (§2.3), and code bpb is read per language group.
- **Markdown cleaning.** Badly converted tables and LaTeX could teach
  noise. The `mdbook` cleaner is checked on statistics and on user spot
  checks, never by quoting text in the repo.

## 8. Experiment log

### C0 — corpus-v3, split-v2, eval-v2, baselines (2026-11-21)

**corpus-v3** (`ig-build-shards`, all §2.3 sources except the
github-code-clean fallback, "keep" books, en and de): 12.56 M documents,
46.4 GB. 19k exact duplicates dropped; shard verification found no
problems.

| source | train | val docs | test docs |
|---|---|---|---|
| code | 13.21 GB | 3,603 | 3,673 |
| web-de | 10.30 GB | 3,306 | 3,385 |
| cosmo | 7.65 GB | 1,763 | 1,730 |
| edu-en | 6.89 GB | 1,461 | 1,376 |
| math | 4.24 GB | 889 | 832 |
| books-md-en | 1.85 GB | 47 books | 44 books |
| data (JSON, YAML) | 1.46 GB | 1,187 | 1,536 |
| tool | 0.36 GB | 174 | 180 |
| books-md-de | 0.29 GB | 23 books | 16 books |

**split-v2** = corpus-v2 + corpus-v1-wiki + corpus-v3: 23.4 M documents,
86.3 GB train.
- Grouped by book, repository and conversation.
- Books are held out at 2% (en) and 6% (de) instead of 1‰, because
  whole books are held out; all other sources stay at 1‰.
- split-v1 rebuilds byte-identically (`--check`).
- All 10.8 M plan-001 documents keep their bucket, source and offset in
  split-v2, so eval-v1 stays clean for runs on split-v2.

**eval-v2**: stratified, 128 chunks × 2048 B per source, 13 sources
("other" and "text" left out), val and test. Val blake2b
`3126e137fe4d4e4a06899f30c82f45a7`.

Existing checkpoints (trained on the old pool), scored by `ig-score`
(eval-v2 val, bpb; KN-5 is trained on a split-v2 train sample):

| source | KN-5 | N2b 300 MB | N2b 900 MB | N2b-L12 900 MB |
|---|---|---|---|---|
| **all (equal weight)** | 2.569 | 1.702 | 1.618 | **1.598** |
| books (Gutenberg) | 2.523 | 1.558 | 1.454 | 1.437 |
| wiki-defs | 2.450 | 1.341 | 1.280 | 1.253 |
| wiki-en | 2.449 | 1.421 | 1.360 | 1.337 |
| wiki-de | 2.636 | 1.454 | 1.381 | 1.362 |
| books-md-en | 2.513 | 1.788 | 1.672 | 1.646 |
| books-md-de | 3.549 | 2.253 | 2.157 | 2.132 |
| edu-en | 2.291 | 1.476 | 1.411 | 1.396 |
| cosmo | 2.145 | 1.497 | 1.434 | 1.415 |
| web-de | 2.574 | 1.734 | 1.669 | 1.655 |
| math | 2.677 | 1.792 | 1.740 | 1.692 |
| code | 2.755 | 2.237 | 2.070 | 2.084 |
| data | 2.780 | 1.949 | 1.882 | 1.848 |
| tool | 2.051 | 1.625 | 1.527 | 1.518 |

- **Weakest where the old pool has nothing:** code (2.07–2.24), the German
  Markdown books (2.13–2.25), data (1.85–1.95), maths (1.69–1.79). The
  plain-prose sources it never saw (edu-en, cosmo) are already close to
  Wikipedia, so modern English prose transfers.
- **German books are the hardest source** for every model and for KN-5
  (3.55). The domain (literature, psychology, practical books) is far
  from wiki-de, and there are only 0.29 GB of them.
- The 12-block model is not better on code (2.084 vs 2.070): depth does
  not help on a domain the data lacks.
- The Markdown books score 0.2 above Gutenberg (1.65 vs 1.44). That fits
  a harder domain (tables, maths, technical text) more than bad cleaning;
  user spot checks of the classes are still open.

### C2-A — M1 with AdamW instead of Lion (2026-11-21)
User question: is the stall of long Lion runs (plan 002 N3c: the 900 MB
run barely moves while lr > 5e-4; WSD's stable phase stalls) a Lion
property, and would AdamW put us on better-documented scaling? M1
protocol, identical data order and seed; only the optimizer differs.
AdamW (`train.optimizer="adamw"`, fp only): lr 2e-3, wd 0.1 (decoupled),
β (0.9, 0.95), grad-norm clip 1.0, same warmup 1000 + cosine to 0.
Untuned (common settings at ~40 M); Lion is the M5-tuned setting.

| eval-v2 val (bpb) | Lion M1 | AdamW M1 | Δ |
|---|---|---|---|
| overall (equal weight) | 1.518 | **1.499** | −0.019 |
| at step 2000 / 4000 / 6000 / 8000 | 1.925 / 1.783 / 1.693 / 1.550 | 1.839 / 1.662 / 1.570 / 1.508 | −0.086 / −0.120 / −0.123 / −0.042 |
| books / wiki-en / wiki-de | 1.628 / 1.500 / 1.570 | 1.594 / 1.477 / 1.535 | −0.033 / −0.023 / −0.035 |
| books-md-en / edu-en / cosmo | 1.422 / 1.404 / 1.130 | 1.408 / 1.386 / 1.093 | −0.014 / −0.018 / −0.037 |
| math / tool | 1.404 / 1.313 | 1.375 / 1.272 | −0.028 / −0.041 |
| code / data (not in M1) | 1.672 / 1.698 | 1.687 / 1.714 | +0.015 / +0.016 |
| books-md-de | 1.830 | 1.830 | 0.000 |
| position eval: window, zero / carried | 1.561 / 1.533 | 1.539 / 1.506 | −0.022 / −0.027 |
| gap closed by the carried state | 32.9% | 37.5% | +4.6 pt |

- **AdamW untuned beats tuned Lion by 0.019** (seed spread ≈ 0.004), on
  10 of 13 sources, and the carried state helps more.
- **The gap is largest mid-run** (−0.12) and shrinks in the cosine tail:
  Lion makes most of its progress only as the lr anneals, consistent
  with a sign-update noise floor.
- Lion transfers slightly better to the sources M1 lacks (code, data);
  C2 M2/M3 will show whether that survives in-domain.
- C2 M2–M4 continue with Lion (user decision), so the mixture ranking
  stays internally consistent. Next, pending user decision: AdamW lr/wd
  sweep on the winning mixture (lr 1e-3 / 4e-3, then wd 0.03 / 0.3, β2
  0.99), then a 900 MB horizon check. Quantization cost under AdamW is
  unmeasured (plan 001 M4/M5 were Lion-only).
