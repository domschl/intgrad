# intgrad

A tiny deep-learning framework in plain C: micrograd-style scalar autograd
where **every value is one byte** (logarithmic quantization), trained
directly on **raw UTF-8 bytes** — no tokenizer.

Goal: a "cognitive core" — a minimal LLM that generates coherent text and
learns tool usage, fitting a CPU + 12 GB VRAM budget. See
[docs/PLAN.md](docs/PLAN.md) for milestones, data strategy, and the
distillation plan.

## Numerics

Every scalar is stored as a single `uint8` in a symmetric log domain
(`include/ig/logquant.h`): sign bit + 7-bit log magnitude with a global
scale. Multiplication is an addition of magnitudes; addition uses the
`log1p(exp)` identity via LUT. Quantization is applied at every forward op
output (quant-aware training), so the network learns to tolerate its own
integer arithmetic.

## Decisions (settled — see docs/PLAN.md §9)

- **D1** Pure-int training is mainline (no float master weights; fp32
  optimizer = last resort with sign-off). Gate: milestone **M5.5**.
- **D2** Tool protocol = control bytes (`include/ig/protocol.h`);
  `ig-pretty` translates streams ⇄ human-readable markup.
- **D3** Teacher = local llama.cpp server at `127.0.0.1:8180`
  (`tools/gen_traces.py`).
- **D4** Latin-family corpus first, then multibyte scripts by adding data;
  ablatable causal byte-conv stem in the M3 architecture.
- **D5** Code: MIT. Data: research-only (provenance manifests anyway).

## Build

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## Try it

```sh
./build/ig-train --data data/sample.txt --steps 2000 --ctx 4 --hidden 64 --sample 200
./build/ig-train --no-quant ...      # exact-float oracle mode (diagnostics)
./build/ig-train --pure-int --grad-stats --gscale 8192 --steps 500
./build/ig-pretty < stream.bin       # control bytes -> ⟪CALL⟫ … markup
./build/ig-pretty -r < markup.txt    # and back
uv run tools/gen_traces.py --n 8 --out data/traces.jsonl --wire data/traces.bin
```

## Data (never committed — see docs/DATA.md)

Starter corpus **corpus-v0** built from local sources (2.7 GB Wikipedia-2017
definitions + 38 GB Project Gutenberg mirror + local en/de TextCorpus):

```sh
uv run tools/build_shards.py --out data/corpus-v0 \
    --wiki-defs data/raw/wikipedia/wiki2017-defs.txt \
    --corpus-dir data/raw/textcorpus \
    --gutenberg-tree data/raw/gutenberg-mirror
# -> 5.3 M protocol-safe, deduped, lang-tagged docs w/ provenance manifest
./build/ig-train --data data/corpus-v0/shard-0000.bin --steps 800 --sample 150
```

## Layout

```
include/ig/     public headers (logquant, value/autograd, net, data)
src/            implementation + ig-train (M2 byte-MLP proof of concept)
tests/          quant round-trip + gradient checks (ctest)
docs/PLAN.md    full project plan: milestones, data, distillation, risks
data/           tiny sample corpus for smoke training
```
