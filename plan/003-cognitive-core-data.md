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
- Weighted batcher (§3), tests for exact resume and for the realised
  weights.

### C2 — Mixture ablations (36 M, 300 MB, seed 42, about 1.4 h each)
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

### C6 — Revisit the nonlinear recurrence (scheduled, agreed 2026-11-21)
Plan 002 N3b was shelved: no gain on the old pool at 36 M.

Code and structured data are where nonlinear state tracking should show
(nesting, indentation, names defined far away). So once C2 has a winner,
rerun N3b on that mixture:
- dense W in the middle block, as before;
- a control: the same layer with W but no tanh;
- scored on eval-v2 code and tool sources and on P5.

Only if that gains does the block-diagonal kernel work from plan 002 come
back.

## 5. Decisions (plan 003)

- **003-D1 Order:** data first, then the best-case fp model, then plan 002
  N4 (user, 2026-11-21).
- **003-D2 Privacy:** every new source lives in data/ under the existing
  privacy rule, whatever its license; the license class is recorded per
  document. No titles or text in the repo.
- **003-D3 Splits:** grouped (book, repository, conversation), so no work
  straddles train and eval.
- **003-D4 Continuity:** eval-v1 is scored on every run, alongside eval-v2.
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

*(empty)*
