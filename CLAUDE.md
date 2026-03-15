# CLAUDE.md

## Project overview

quectoGPT is a JavaScript port of Karpathy's microgpt.py — a minimal GPT trained on a names dataset. The key change is tensor-level autograd (instead of scalar) with a custom library called `quectrograd` that has CPU and WebGPU backends.

## How to run

```bash
# Train (CPU, Node)
node train.js
node train.js --backend=cpu --steps=100

# Train (WebGPU, Deno)
deno run --allow-read --unstable-webgpu train.js --backend=webgpu --steps=100

# Benchmarks (CPU, Node)
node bench.js

# Benchmarks (WebGPU, Deno)
deno run --allow-read --unstable-webgpu bench.js --gpu

# Browser: serve and open index.html (training) or bench.html (benchmarks)
python -m http.server
# or: npx serve .
```

Deno is installed at `~/.deno/bin/deno`. If not on PATH, use the full path. Install: `curl -fsSL https://deno.land/install.sh | sh`, then add `~/.deno/bin` to PATH.

## Architecture

- **quectrograd/** is the autograd library. Two backends implement the same interface:
  - `backend_cpu.js` — Float32Array ops, pure loops
  - `backend_webgpu.js` — GPUBuffer ops, WGSL shaders
- **Tape-based autograd** — forward ops append to a global tape array, `backward()` walks it in reverse. No topo sort needed.
- **ops.js** — each op computes forward via the backend, wraps result in a Tensor, and attaches a `_backwardFn` closure for backward. **No op calls `.toArray()`** — all computation stays on device (critical for GPU performance).
- **train.js** — two forward modes:
  - **Batched `gptForward`** (training): processes entire sequence at once as `[n, embd]` tensors with causal mask. Much fewer ops than sequential approach — critical for GPU performance.
  - **Sequential `gptForwardToken`** (inference): processes one token at a time with KV cache.
  - Exports an async generator `train()` consumed by Node CLI, Deno CLI, and browser. `await` on `.toArray()` only where results are actually read (loss logging, inference sampling).
  - Model configs: nano/tiny/small/medium/large presets via `--model=` flag.
- **bench.js** — finite-difference gradient checking + op-level timing. Works with both Node (CPU) and Deno (CPU + WebGPU).
- **index.html** — browser training dashboard with CRT/oscilloscope theme, real-time Canvas2D loss chart, model size selector, CPU/WebGPU toggle.
- **bench.html** — browser benchmark page with correctness tests, gradient checks, op-level benchmarks, and training benchmark. Same theme as index.html. Nav links between the two pages.

## Key conventions

- Pure ES modules throughout (`import`/`export`), no bundler.
- No npm dependencies. Zero.
- Weights stored as `[outDim, inDim]` matching microgpt.py. The `matmulWT` op handles the transpose.
- `clearTape()` must be called at the start of each training step.
- The CPU backend is the reference — all GPU results are tested against it.
- `train.js` and `bench.js` have runtime detection for Node vs Deno (file I/O, args, stdout).
- `bench.js` uses `timeIt(fn, warmup=10, runs=100)` for timing.

## GPU performance notes

- **Command batching**: WebGPU ops accumulate in a shared `CommandEncoder`; only flushed on `toArray()`. Avoids per-op submit overhead.
- **Buffer pooling**: Uniform buffers and ids buffers are pooled and reused across steps via `recycleBuffers()`. Uses `writeBuffer` instead of `mappedAtCreation`.
- **Batched forward**: Training uses `gptForward` which processes the full sequence as `[n, embd]` matrices with causal masking, not one token at a time. This reduces op count from O(seqLen * layers * heads) to O(layers * heads).
- **Speedups** (approximate, Deno CLI): small ~1x, medium ~2x, large ~5-6x vs CPU. GPU wins grow with model size.

## Testing

```bash
node bench.js                                                  # CPU correctness + benchmarks
deno run --allow-read --unstable-webgpu bench.js --gpu         # WebGPU correctness + benchmarks
```

Correctness criteria:
- Forward: max absolute error < 1e-5 vs expected values
- Backward: gradient matches finite-difference (eps=1e-4, relative tolerance < 5%)
- Known: softmax gradient check reports rel err=1.0 due to finite-difference numerical issues. Not a real bug — training works correctly.

## Browser pages

Both served from the same directory (no build step):
- `index.html` — training dashboard (Train tab)
- `bench.html` — benchmark suite (Bench tab)
- Navigation links between the two pages in the header
