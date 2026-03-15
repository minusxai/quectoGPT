# CLAUDE.md

## Project overview

quectoGPT is a JavaScript port of Karpathy's microgpt.py — a minimal GPT trained on names or Shakespeare datasets. Uses [jax-js](https://github.com/ekzhang/jax-js) (`@jax-js/jax` + `@jax-js/optax`) for tensor operations, autodiff (`valueAndGrad`), and optimization (`adam`). Supports Wasm (CPU) and WebGPU backends. Byte-pair encoding (BPE) tokenizer trained per-dataset.

## How to run

**IMPORTANT: Always use Deno to run JS files, NEVER use node.** Deno is installed at `~/.deno/bin/deno`. If not on PATH, use the full path. Install: `curl -fsSL https://deno.land/install.sh | sh`, then add `~/.deno/bin` to PATH.

```bash
# 1. Train tokenizer (one-time per dataset)
deno run --allow-read --allow-write --allow-env tokenize.js --dataset=names --vocab=1024
deno run --allow-read --allow-write --allow-env tokenize.js --dataset=shakespeare --vocab=1024

# 2. Train model (CPU/Wasm)
deno run --allow-read --allow-net --allow-env train.js --dataset=names --model=nano --steps=100
deno run --allow-read --allow-net --allow-env train.js --dataset=shakespeare --model=small --steps=200 --blocksize=128 --prompt="To be"

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
- **bpe.js** — BPE tokenizer module:
  - `trainBPE(textBytes, numMerges)` — learns merge rules from raw bytes
  - `buildBPETokenizer(merges)` — returns `{ encode, decode, vocabSize, BOS, EOS }` from a merges list
- **tokenize.js** — CLI script for BPE training + data pre-processing:
  - Trains BPE on `data/<dataset>.txt`
  - Splits into train/val (90/10 by lines)
  - Outputs `data/<dataset>.tok.json` (merges + metadata), `data/<dataset>.train.bin`, `data/<dataset>.val.bin` (Uint16Array binary)
  - Prints metrics: compression ratio, bytes/token, top merges, roundtrip check
- **gpt.js** — model definition as pure functions:
  - `MODEL_CONFIGS` — nano/tiny/small/medium/large presets
  - `initParams(vocabSize, cfg, rngKey)` — returns nested pytree `{ wte, wpe, lmHead, layers: [{wq, wk, wv, wo, mlpFc1, mlpFc2}, ...] }`
  - `forward(params, cfg, tokenOH, posOH, seqLen, batched)` — returns logits. Supports both unbatched `[seqLen, vocabSize]` and batched `[B, seqLen, vocabSize]`.
  - `loss(params, cfg, tokenOH, posOH, targetOH, seqLen, mask, batched)` — cross-entropy loss scalar with optional mask for padding
  - `inference(params, cfg, tokenizer, rngKey, opts)` — sequential generation with optional `opts.prompt` for seeded continuation
- **train.js** — training loop:
  - `train(backendName, trainData, valData, tokenizer, opts)` — async generator yielding `{init, step, done}` events
  - Loads pre-tokenized Uint16Array binary data (no on-the-fly tokenization)
  - Window-based sampling: random `blockSize`-length windows from flat token stream
  - Batched training (default batch size 8) — all sequences same length, no padding needed
  - Val loss evaluated periodically (~20 times during training)
  - Uses `valueAndGrad(lossFn)(params)` for forward+backward in one call
  - optax `adam` with 3-phase LR schedule (warmup → constant → decay)
  - CLI flags: `--dataset`, `--model`, `--backend`, `--steps`, `--batch`, `--blocksize`, `--prompt`
- **bench.js** — correctness tests + benchmarks using jax-js `grad()` for gradient checks
- **index.html** — browser training dashboard with CRT/oscilloscope theme, real-time Canvas2D loss chart (train=green, val=red), dataset selector, model size selector, CPU/WebGPU toggle, prompt input.
- **bench.html** — browser benchmark page with correctness tests, gradient checks, op-level benchmarks, and training benchmark. Same theme as index.html. Nav links between the two pages.

## Data pipeline

```
data/<dataset>.txt          — raw text (one name per line, or full Shakespeare text)
    ↓ tokenize.js --dataset=<name> --vocab=1024
data/<dataset>.tok.json     — BPE merges + metadata (vocab_size, bos, eos, metrics)
data/<dataset>.train.bin    — pre-tokenized train split (Uint16Array, flat token stream)
data/<dataset>.val.bin      — pre-tokenized val split (Uint16Array, flat token stream)
    ↓ train.js --dataset=<name>
    loads .tok.json → buildBPETokenizer(merges) for inference
    loads .train.bin / .val.bin → random window sampling for batches
```

## Key conventions

- Pure ES modules throughout (`import`/`export`), no bundler.
- npm dependencies: `@jax-js/jax` and `@jax-js/optax`. Browser uses esm.sh CDN via import maps.
- Weights stored as `[inDim, outDim]`. Forward uses `np.dot(x, w)` — no transpose needed.
- Reference counting: use `.ref` when an array is consumed by multiple ops. Use `tree.ref(params)` to ref all leaves.
- `tree.dispose()` to free arrays when done.
- Params are a nested pytree — `valueAndGrad` differentiates w.r.t. the first argument automatically.
- `train.js` and `bench.js` have runtime detection for Deno vs other runtimes (file I/O, args, stdout).
- Backend init: `await init()` returns available devices, then `defaultDevice("wasm"` or `"webgpu")`.
- BPE tokenizer: 256 byte tokens + N merges + BOS + EOS. Merges stored as `[[a, b], ...]` pairs of token IDs in merge order.

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
- `index.html` — training dashboard (Train tab) — dataset selector, prompt input, train+val loss chart
- `bench.html` — benchmark suite (Bench tab)
- Navigation links between the two pages in the header
