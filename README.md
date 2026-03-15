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
- **optax `adam`** with LR schedule (warmup → constant → linear decay)
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

### Node.js (CPU via Wasm)

```bash
node train.js                                    # nano, 100 steps
node train.js --model=small --steps=200          # small model
node train.js --model=medium                     # medium, uses config defaults
```

### Deno (WebGPU)

Deno has built-in WebGPU support — the easiest way to run GPU from CLI:

```bash
# Install Deno (if needed)
curl -fsSL https://deno.land/install.sh | sh

# Train on GPU
deno run --allow-read --allow-net --unstable-webgpu train.js --backend=webgpu --model=small
deno run --allow-read --allow-net --unstable-webgpu train.js --backend=webgpu --model=medium --steps=100

# GPU benchmarks
deno run --allow-read --allow-net --unstable-webgpu bench.js --gpu

# CPU benchmarks
deno run --allow-read --allow-net bench.js --cpu
```

### Benchmarks (Node)

```bash
node bench.js           # CPU correctness + benchmarks (default)
node bench.js --cpu     # explicit CPU
```

### Browser

Serve the directory and open `index.html` or `bench.html`:
```bash
python -m http.server
# or: npx serve .
```

No build step — pure ES modules.

- **index.html** — training dashboard with real-time loss chart. Auto-detects WebGPU. Model size selector (nano through large), configurable steps, backend toggle.
- **bench.html** — correctness tests, gradient checks, op-level benchmarks, training benchmark. Supports CPU (Wasm) and WebGPU.

## Requirements

- **Node.js 18+** for CPU (Wasm) backend
- **Deno** for WebGPU from CLI (`curl -fsSL https://deno.land/install.sh | sh`)
- **Any modern browser** with WebGPU support (Chrome 113+, Edge 113+) for GPU in browser
- npm dependencies: `@jax-js/jax` and `@jax-js/optax` (`npm install`). Browser uses esm.sh CDN.
