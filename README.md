# quectoGPT

The most atomic way to train a GPT — now in JavaScript with WebGPU acceleration.

A port of Karpathy's [microgpt.py](https://github.com/karpathy/microgpt) using [jax-js](https://github.com/ekzhang/jax-js) (`@jax-js/jax` + `@jax-js/optax`) — a production-grade ML library with JAX API, WebGPU/Wasm backends, built-in `grad()`, `jit()`, and optimizers.

## What this is

- **bpe.js** — byte-pair encoding tokenizer (training + encode/decode)
- **tokenize.js** — CLI script: train BPE, split train/val, save binary token files
- **gpt.js** — GPT model definition: config, param init, forward, loss, inference (pure functions)
- **train.js** — batched training loop with `valueAndGrad` + optax `adam`, train/val loss tracking
- **bench.js** — correctness tests + benchmarks using jax-js ops + `grad()`
- **index.html** — browser training dashboard with real-time loss chart (train + val)
- **bench.html** — browser benchmark page (correctness tests, gradient checks, op timing, training bench)

## Architecture

```
quectoGPT/
  data/
    names.txt             — names dataset (32K names, one per line)
    shakespeare.txt       — Shakespeare text (40K lines)
    *.tok.json            — trained BPE tokenizer (merges + metadata)
    *.train.bin           — pre-tokenized train split (Uint16Array)
    *.val.bin             — pre-tokenized val split (Uint16Array)
  bpe.js                  — BPE tokenizer: trainBPE, buildBPETokenizer
  tokenize.js             — tokenizer training + data pre-processing CLI
  gpt.js                  — model: configs, initParams, forward, loss, inference
  train.js                — training loop: window sampling, batched valueAndGrad, adam
  bench.js                — correctness tests + benchmarks (CLI)
  index.html              — browser training dashboard
  bench.html              — browser benchmark page
```

Key patterns:
- **BPE tokenizer** — byte-level BPE (1024 vocab). Trained per-dataset, saves merges as JSON. Data pre-tokenized into binary files for fast loading.
- **Window-based sampling** — random `blockSize`-length windows from flat token stream. No padding, no masking.
- **Batched training** — configurable batch size (default 8). All sequences same length within batch.
- **`valueAndGrad`** computes loss and gradients in one call
- **optax `adam`** with LR schedule (warmup → constant → linear decay)
- **`nn.dotProductAttention`** with `isCausal: true` — no manual per-head loop
- **Reference counting** via `.ref` for arrays consumed by multiple ops
- **Prompted inference** — seed with text for continuation (e.g., Shakespeare prompts)
- Params stored as nested pytree `{ wte, wpe, lmHead, layers: [{wq, wk, wv, wo, mlpFc1, mlpFc2}, ...] }`

## Model configs

Five preset sizes, selected via `--model=`:

| Config | Layers | Embed | Heads | Context | Params | Default LR |
|--------|--------|-------|-------|---------|--------|------------|
| `nano` | 1 | 16 | 4 | 16 | ~4K | 0.01 |
| `tiny` | 2 | 64 | 4 | 32 | ~100K | 3e-3 |
| `small` | 4 | 128 | 4 | 64 | ~500K | 1e-3 |
| `medium` | 6 | 256 | 8 | 128 | ~4M | 3e-4 |
| `large` | 8 | 512 | 8 | 256 | ~25M | 1e-4 |

All configs: BPE tokenizer (1024 vocab), RMSNorm, ReLU, no biases. Context window (`blockSize`) overridable via `--blocksize=`.

## Quick start

### Install dependencies

```bash
npm install
```

### Train tokenizer (one-time per dataset)

```bash
deno run --allow-read --allow-write --allow-env tokenize.js --dataset=names --vocab=1024
deno run --allow-read --allow-write --allow-env tokenize.js --dataset=shakespeare --vocab=1024
```

### Train model (CPU via Wasm)

```bash
# Names dataset
deno run --allow-read --allow-net --allow-env train.js --dataset=names --model=tiny --steps=200

# Shakespeare with larger context + prompted inference
deno run --allow-read --allow-net --allow-env train.js --dataset=shakespeare --model=small --steps=200 --blocksize=128 --prompt="To be"
```

### Train (WebGPU — browser only)

jax-js's WebGPU backend requires `OffscreenCanvas`, which is only available in browsers (not Deno CLI). Open `index.html` in Chrome for GPU training.

### Benchmarks

```bash
# CPU (Wasm)
deno run --allow-read --allow-net --allow-env bench.js

# WebGPU
deno run --allow-read --allow-net --allow-env --unstable-webgpu bench.js --gpu
```

### Browser

Serve the directory and open `index.html` or `bench.html`:
```bash
python -m http.server
# or: npx serve .
```

No build step — pure ES modules. Browser uses esm.sh CDN for jax-js.

- **index.html** — training dashboard with real-time train+val loss chart. Dataset selector (names/shakespeare), model size, prompt input, configurable steps, backend toggle.
- **bench.html** — correctness tests, gradient checks, op-level benchmarks, training benchmark. Supports CPU (Wasm) and WebGPU.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset=` | `names` | Dataset to train on (`names`, `shakespeare`) |
| `--model=` | `nano` | Model size preset |
| `--steps=` | per-model | Number of training steps |
| `--batch=` | `8` | Batch size |
| `--blocksize=` | per-model | Context window size override |
| `--backend=` | `cpu` | Backend (`cpu` or `webgpu`) |
| `--prompt=` | _(none)_ | Seed text for inference |

## Requirements

- **Deno** — runtime for CLI (`curl -fsSL https://deno.land/install.sh | sh`)
- **Any modern browser** with WebGPU support (Chrome 113+, Edge 113+) for GPU in browser
- npm dependencies: `@jax-js/jax` and `@jax-js/optax` (`npm install`)
