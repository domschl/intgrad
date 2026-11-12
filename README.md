# intgrad

A tiny deep-learning framework in plain C: micrograd-style scalar autograd
where **every value is one byte** (logarithmic quantization), trained
directly on **raw UTF-8 bytes** — no tokenizer.

Goal: a "cognitive core" — a minimal LLM that generates coherent text and
learns tool usage, fitting a CPU + 12 GB VRAM budget. See
[docs/PLAN.md](docs/PLAN.md) for milestones, data strategy, and the
distillation plan.

## Numerics

Every scalar the network computes with is a single `uint8` in a symmetric log domain
(`include/ig/logquant.h`): sign bit + 7-bit log magnitude with a global
scale. Multiplication is an addition of magnitudes; addition uses the
`log1p(exp)` identity via LUT. Quantization is applied at every forward op
output (quant-aware training), so the network learns to tolerate its own
integer arithmetic. Dot products accumulate exactly and are requantized
once per output (int32-GEMM semantics).

**Parameters are stored as two bytes** (decision D7, C1 amended): a
16-bit master log code whose high byte is the 1-byte weight the forward
and backward passes use; the low byte accumulates optimizer updates that
are smaller than one byte-grid step (7.2%) — without it ~98% of updates
are lost (E7), and 1-byte stochastic rounding drifts +0.15 bpb over long
runs (E8). Two-byte weights stay within ~0.01–0.03 bpb of float.

The default optimizer is **Lion** (decision D8): a single momentum
stored as 16-bit log codes and a sign update — no square root, division
or epsilon, so its integer form is a sign test plus one code comparison.

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
- **D6** Hybrid grids: log-byte weights/activations, fixed-point byte
  logits, exact accumulation + one requant per output, clamp-aware STE.
- **D7** Two-byte parameter storage (C1 amended): forward reads the high
  byte, the low byte accumulates updates (round-to-nearest by default).
- **D8** Lion is the default optimizer (one int-native 16-bit momentum,
  sign update; lr 2e-3, wd 0.1, batch 32); Adam via `--opt adam`.
  Accepted cost: ~0.1 bpb vs Adam at matched data on the M2 vehicle (E9).

## Build

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## Try it

```sh
./build/ig-train --data data/sample.txt --steps 500 --ctx 4 --hidden 64 --sample 200
./build/ig-train --no-quant ...      # exact-float oracle mode (diagnostics)
./build/ig-train --pure-int --grad-stats --steps 500   # dyn per-tensor gscale
./build/ig-train --opt adam ...      # Adam instead of Lion (defaults lr 3e-3, batch 1, wd 0)
./build/ig-train --w2-sr ...         # stochastic rounding of the low weight byte
./build/ig-train --w1 ...            # 1-byte weight storage + SR (ablation, E7/E8)
./build/ig-train --w1 --no-sr ...    # 1-byte round-to-nearest (the old bug, E7)
./build/ig-pretty < stream.bin       # control bytes -> ⟪CALL⟫ … markup
./build/ig-pretty -r < markup.txt    # and back
uv run tools/gen_traces.py --n 8 --out data/traces.jsonl --wire data/traces.bin
```

## Training & resume semantics

`--steps` is the **total horizon** (absolute), `--save` writes checkpoints
containing: params (fp32 + byte code + 16-bit master codes), **optimizer
state** (Lion momentum or Adam moments), **steps done**,
and **RNG position** — so `--load` continues the LR schedule and data order
bit-exactly (guarded by `ctest` checkpoint test; atomic tmp+rename writes,
`--save-every N` for crash-safe periodic saves). Hyperparameters (lr, batch,
arch, warmup, sched) are *not* in the checkpoint: pass the same flags on
resume — including `--opt` (the trainer warns if a checkpoint's optimizer
state doesn't match). Note a cosine run of N steps is not a prefix of a longer horizon —
always give every segment the same total `--steps`.

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
