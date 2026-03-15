# quectoGPT

The most atomic way to train a GPT — now in JavaScript with WebGPU acceleration.

A port of Karpathy's [microgpt.py](https://github.com/karpathy/microgpt) using [jax-js](https://github.com/ekzhang/jax-js) (`@jax-js/jax` + `@jax-js/optax`) — a production-grade ML library with JAX API, WebGPU/Wasm backends, built-in `grad()`, `jit()`, and optimizers.

## What this is

- **gpt.js** — GPT model definition: config, param init, forward, loss, inference (pure functions)
- **train.js** — training loop with `valueAndGrad` + optax `adam`
- **bench.js** — correctness tests + benchmarks using jax-js ops + `grad()`
- **index.html** — browser training dashboard with real-time loss chart
- **bench.html** — browser benchmark page (correctness tests, gradient checks, op timing, training bench)

## Architecture

The model (`gpt.js`) is a collection of pure functions operating on a params pytree:

```
quectoGPT/
  node_modules/         — @jax-js/jax + @jax-js/optax
  gpt.js                — model: configs, initParams, forward, loss, inference
  train.js              — training loop: tokenizer, valueAndGrad, adam optimizer
  bench.js              — correctness tests + benchmarks (CLI)
  index.html            — browser training dashboard
  bench.html            — browser benchmark page
  input.txt             — training data (names dataset)
```

Key patterns:
- **`valueAndGrad`** computes loss and gradients in one call
- **optax `adam`** with LR schedule (warmup -> constant -> linear decay)
- **`nn.dotProductAttention`** with `isCausal: true` — no manual per-head loop
- **Reference counting** via `.ref` for arrays consumed by multiple ops
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

All configs: char-level tokenizer (27 tokens: a-z + BOS), RMSNorm, ReLU, no biases.

## Quick start

### Install dependencies

```bash
npm install
```

### Train (CPU via Wasm)

```bash
deno run --allow-read --allow-net --allow-env train.js                         # nano, 100 steps
deno run --allow-read --allow-net --allow-env train.js --model=small --steps=200
deno run --allow-read --allow-net --allow-env train.js --model=medium
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
uv run python -m http.server
# or: npx serve .
```

No build step — pure ES modules. Browser uses esm.sh CDN for jax-js.

- **index.html** — training dashboard with real-time loss chart. Auto-detects WebGPU. Model size selector (nano through large), configurable steps, backend toggle.
- **bench.html** — correctness tests, gradient checks, op-level benchmarks, training benchmark. Supports CPU (Wasm) and WebGPU.

## Requirements

- **Deno** — runtime for CLI (`curl -fsSL https://deno.land/install.sh | sh`)
- **Any modern browser** with WebGPU support (Chrome 113+, Edge 113+) for GPU in browser
- npm dependencies: `@jax-js/jax` and `@jax-js/optax` (`npm install`)
