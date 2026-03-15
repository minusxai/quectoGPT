# quectoGPT

The most atomic way to train a GPT — now in JavaScript with WebGPU acceleration.

A port of Karpathy's [microgpt.py](https://github.com/karpathy/microgpt) from scalar autograd to **tensor-level autograd**, with a custom tensor library (`quectrograd`) that runs on both CPU (pure JS) and WebGPU (WGSL compute shaders).

## What this is

- **quectrograd/** — a minimal tensor autograd library with two backends:
  - `backend_cpu.js` — pure JS, no dependencies, works everywhere
  - `backend_webgpu.js` — WebGPU accelerated via WGSL compute shaders
- **train.js** — GPT model definition + training loop (works with either backend)
- **bench.js** — correctness tests + CPU vs WebGPU benchmarks
- **index.html** — browser training dashboard with real-time loss chart
- **bench.html** — browser benchmark page (correctness tests, gradient checks, op timing, training bench)

## Architecture

The key shift from the Python version: each `Value` node held a single float. Here, each `Tensor` holds a `Float32Array` (CPU) or `GPUBuffer` (WebGPU), and ops dispatch at the tensor level. Autograd uses a **tape** (flat array) instead of graph traversal — forward ops append to the tape, backward walks it in reverse.

```
quectoGPT/
  quectrograd/
    backend_cpu.js    — pure JS backend (reference implementation)
    backend_webgpu.js — WebGPU backend (WGSL shaders inline)
    tensor.js         — Tensor class
    ops.js            — forward+backward op wrappers
    autograd.js       — tape-based backward pass
    optim.js          — Adam optimizer
    index.js          — public API
  train.js            — GPT model + training loop
  bench.js            — correctness tests + benchmarks (CLI)
  index.html          — browser training dashboard
  bench.html          — browser benchmark page
  input.txt           — training data (names dataset)
```

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

### Node.js (CPU)

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
deno run --allow-read --unstable-webgpu train.js --backend=webgpu --model=small
deno run --allow-read --unstable-webgpu train.js --backend=webgpu --model=medium --steps=100

# GPU benchmarks
deno run --allow-read --unstable-webgpu bench.js --gpu

# CPU benchmarks
deno run --allow-read bench.js --cpu
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
- **bench.html** — correctness tests, gradient checks, op-level benchmarks, training benchmark. Supports CPU and WebGPU.

## Performance

GPU speedup grows with model size (approximate, Deno CLI on Apple M-series):

| Model | Params | CPU ms/step | GPU ms/step | Speedup |
|-------|--------|-------------|-------------|---------|
| small | 800K | 45 | 57 | ~1x |
| medium | 4.8M | 258 | 126 | **2x** |
| large | 25M | 1,380 | 245 | **5.6x** |

Key optimizations:
- **Batched sequence processing**: training forward pass processes all tokens at once as `[seqLen, embd]` matrices with causal masking, instead of one-at-a-time with KV cache
- **GPU command batching**: ops accumulate in a shared `CommandEncoder`, flushed only on readback
- **Buffer pooling**: uniform and ids buffers reused across training steps

## Requirements

- **Node.js 18+** for CPU backend
- **Deno** for WebGPU from CLI (`curl -fsSL https://deno.land/install.sh | sh`)
- **Any modern browser** with WebGPU support (Chrome 113+, Edge 113+) for GPU in browser
- No npm dependencies. Zero. Just JS files.
