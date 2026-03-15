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

## Quick start

### Node.js (CPU)

```bash
node train.js
node train.js --backend=cpu --steps=100
```

### Deno (WebGPU)

Deno has built-in WebGPU support — this is the easiest way to run GPU from the CLI:

```bash
# Train on GPU
deno run --allow-read --unstable-webgpu train.js --backend=webgpu --steps=100

# GPU benchmarks
deno run --allow-read --unstable-webgpu bench.js --gpu

# CPU benchmarks
deno run --allow-read bench.js --cpu
```

### Benchmarks (Node)

```bash
node bench.js           # CPU correctness + benchmarks (default)
node bench.js --cpu     # explicit CPU
node bench.js --all     # CPU + GPU (GPU needs Deno, see above)
```

### Browser

Serve the directory and open `index.html` or `bench.html`:
```bash
npx serve .
# or
python -m http.server
```

No build step — pure ES modules.

- **index.html** — training dashboard with real-time loss chart. Auto-detects WebGPU. Configurable steps + backend toggle.
- **bench.html** — runs correctness tests, gradient checks, op-level benchmarks, and a training benchmark in-browser. Supports CPU and WebGPU backends.

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
- **Deno** for WebGPU from CLI (`curl -fsSL https://deno.land/install.sh | sh`)
- **Any modern browser** with WebGPU support (Chrome 113+, Edge 113+) for GPU in browser
- No npm dependencies. Zero. Just JS files.
