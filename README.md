# intgrad

A toy byte-level language model with **byte numerics**:

- **No tokenizer.** The model reads raw UTF-8 bytes and writes raw UTF-8
  bytes (vocabulary = 256).
- **1-byte compute, 1–2-byte weights.** Every weight, activation and logit
  the network computes with is one byte on a logarithmic (symlog) grid.
  Logits use a fixed-point byte grid. Parameters are stored as a 16-bit
  master code whose high byte is the compute weight (default), or as one
  byte with stochastic rounding.
- **Lion** is the optimizer: one 16-bit momentum and a sign update, with
  no sqrt, division or epsilon.

**Current plan: [plan/001](plan/001-reset-byte-transformer.md)**: review,
test methodology, milestones (data batchers → PyTorch reference engine →
scalable test transformer → quantized transformer → recurrence readiness).
Plans are numbered, and later ones build on earlier ones.

## What is in the repo now

The C library is the **numerical oracle**: a micrograd-style scalar
autograd with the exact quantization semantics, a byte-MLP trainer
(`ig-train`), and regression tests. The scalable transformer is built in
a PyTorch reference engine (plan/001 M2–M3), bound to the oracle by parity
tests.

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build

./build/ig-train --data tests/fixtures/sample.txt --steps 500 --hidden 64 --sample 200
./build/ig-train --index data/split-v1/idx.bin --steps 2500 --hidden 96 \
    --eval-every 500 --eval-windows 800
#   --no-quant   exact float oracle      --pure-int  byte-grid gradients
#   --w1         1-byte weights + SR     --w2-sr     SR on the low weight byte
```

The Python package in `py/` (uv) holds the corpus pipeline, the batchers
and the eval set (and, from plan/001 M2, the PyTorch reference engine):

```sh
cd py && uv sync --extra corpus
uv run pytest                                    # unit tests
uv run python -m intgrad.data --index ../data/split-v1/idx.bin   # batcher throughput
```

`--steps` is the total horizon. `--save`/`--load` resume bit-exactly
(parameters, master codes, Lion momentum, data and SR RNG); pass the same
flags on resume.

## Data is private

`data/` and `runs/` are **never committed** (`.gitignore`, no exceptions).
Much of the corpus is not openly licensed. The repo contains only the
pipeline (`py/intgrad/corpus/`; see [docs/DATA.md](docs/DATA.md)). Small fixtures written for this repo live in
`tests/fixtures/`.

## Layout

```
plan/           numbered plans (001 = current)
include/ src/   C oracle: logquant (grids), value (autograd), net (MLP, Lion), dataset
tests/          ctest: quantizer, autograd, checkpoint resume, training numerics, sampler
py/             uv project `intgrad`: corpus pipeline (Gutenberg cleaner, shards,
                split/index v2), batchers (random + sequential), eval set + baselines
docs/           DATA.md; history/ = frozen v0 plan and experiment log
```

License: MIT (code). Training data: not distributed.
