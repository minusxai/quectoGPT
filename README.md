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
  bench.js            — correctness tests + benchmarks
  index.html          — browser training dashboard
  input.txt           — training data (names dataset)
```

## Quick start

### Node.js (CPU)

```bash
node train.js
```

Options:
```bash
node train.js --backend=cpu --steps=100
```

### Benchmarks

```bash
node bench.js           # CPU correctness + benchmarks
node bench.js --gpu     # also test WebGPU (requires Node 22+ with --experimental-webgpu)
```

### Browser

Serve the directory and open `index.html`:
```bash
npx serve .
```

Or any static file server. No build step — pure ES modules.

The dashboard auto-detects WebGPU. You'll see a real-time loss chart as the model trains on names, then generated samples when done.

## Model

Same architecture as microgpt.py — a tiny GPT-2 variant:
- 1 transformer layer
- 16-dim embeddings
- 4 attention heads
- 16 token context window
- ~4K parameters
- Char-level tokenizer (27 tokens: a-z + BOS)
- RMSNorm (not LayerNorm), ReLU (not GELU), no biases

## Requirements

- **Node.js 18+** for CPU backend
- **Node.js 22+** with `--experimental-webgpu` for WebGPU backend
- **Any modern browser** with WebGPU support (Chrome 113+, Edge 113+, Firefox Nightly) for GPU in browser
- No npm dependencies. Zero. Just JS files.
