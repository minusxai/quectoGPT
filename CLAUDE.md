# CLAUDE.md

## Project overview

quectoGPT is a JavaScript port of Karpathy's microgpt.py — a minimal GPT trained on a names dataset. Uses [jax-js](https://github.com/ekzhang/jax-js) (`@jax-js/jax` + `@jax-js/optax`) for tensor operations, autodiff (`valueAndGrad`), and optimization (`adam`). Supports Wasm (CPU) and WebGPU backends.

## How to run

**IMPORTANT: Always use Deno to run JS files, NEVER use node.** Deno is installed at `~/.deno/bin/deno`. If not on PATH, use the full path. Install: `curl -fsSL https://deno.land/install.sh | sh`, then add `~/.deno/bin` to PATH.

```bash
# Train (CPU/Wasm)
deno run --allow-read --allow-net --allow-env train.js
deno run --allow-read --allow-net --allow-env train.js --backend=cpu --steps=100

# Train (WebGPU — browser only, jax-js requires OffscreenCanvas)
# Open index.html in Chrome for GPU training

# Benchmarks (CPU/Wasm)
deno run --allow-read --allow-net --allow-env bench.js

# Benchmarks (WebGPU — browser only, jax-js requires OffscreenCanvas)
# Open bench.html in Chrome for GPU benchmarks

# Browser: serve and open index.html (training) or bench.html (benchmarks)
python -m http.server
# or: npx serve .
```

## Architecture

- **@jax-js/jax + @jax-js/optax** (npm packages). Provides `numpy`, `nn`, `grad`, `valueAndGrad`, `random`, `tree`, and `optax` (adam optimizer). Backends: `"wasm"` (fast, default), `"webgpu"` (GPU, browser only — requires OffscreenCanvas which Deno lacks), `"cpu"` (slow JS interpreter for debugging).
- **gpt.js** — model definition as pure functions:
  - `MODEL_CONFIGS` — nano/tiny/small/medium/large presets
  - `initParams(vocabSize, cfg, rngKey)` — returns nested pytree `{ wte, wpe, lmHead, layers: [{wq, wk, wv, wo, mlpFc1, mlpFc2}, ...] }`
  - `forward(params, cfg, inputIds, posIds)` — returns logits `[seqLen, vocabSize]`
  - `loss(params, cfg, inputIds, posIds, targetIds, vocabSize)` — cross-entropy loss scalar
  - `inference(params, cfg, tokenizer, rngKey, opts)` — sequential generation
- **train.js** — training loop:
  - `buildTokenizer(docs)` — char-level tokenizer (a-z + BOS)
  - `train(backendName, docs, opts)` — async generator yielding `{init, step, done}` events
  - Uses `valueAndGrad(lossFn)(params)` for forward+backward in one call
  - optax `adam` with 3-phase LR schedule (warmup → constant → decay)
  - Backend mapping: `--backend=cpu` → `"wasm"`, `--backend=webgpu` → `"webgpu"`
- **bench.js** — correctness tests + benchmarks using jax-js `grad()` for gradient checks
- **index.html** — browser training dashboard with CRT/oscilloscope theme, real-time Canvas2D loss chart, model size selector, CPU/WebGPU toggle.
- **bench.html** — browser benchmark page with correctness tests, gradient checks, op-level benchmarks, and training benchmark. Same theme as index.html. Nav links between the two pages.

## Key conventions

- Pure ES modules throughout (`import`/`export`), no bundler.
- npm dependencies: `@jax-js/jax` and `@jax-js/optax`. Browser uses esm.sh CDN via import maps.
- Weights stored as `[inDim, outDim]`. Forward uses `np.dot(x, w)` — no transpose needed.
- Reference counting: use `.ref` when an array is consumed by multiple ops. Use `tree.ref(params)` to ref all leaves.
- `tree.dispose()` to free arrays when done.
- Params are a nested pytree — `valueAndGrad` differentiates w.r.t. the first argument automatically.
- `train.js` and `bench.js` have runtime detection for Deno vs other runtimes (file I/O, args, stdout).
- Backend init: `await init()` returns available devices, then `defaultDevice("wasm"` or `"webgpu")`.

## Testing

```bash
deno run --allow-read --allow-net --allow-env bench.js                                  # CPU (wasm) correctness + benchmarks
deno run --allow-read --allow-net --allow-env --unstable-webgpu bench.js --gpu           # WebGPU correctness + benchmarks
```

Correctness criteria:
- Forward: max absolute error < 1e-5 vs expected values
- Backward: gradient via `grad()` matches finite-difference (eps=1e-4, relative tolerance < 5%)

## Browser pages

Both served from the same directory (no build step):
- `index.html` — training dashboard (Train tab)
- `bench.html` — benchmark suite (Bench tab)
- Navigation links between the two pages in the header
