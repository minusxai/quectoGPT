# CLAUDE.md

## Project overview

quectoGPT is a JavaScript port of Karpathy's microgpt.py — a minimal GPT trained on a names dataset. The key change is tensor-level autograd (instead of scalar) with a custom library called `quectrograd` that has CPU and WebGPU backends.

## How to run

```bash
# Train (CPU)
node train.js

# Train with options
node train.js --backend=cpu --steps=100

# Run correctness tests + benchmarks
node bench.js

# Browser: serve and open index.html
npx serve .
```

## Architecture

- **quectrograd/** is the autograd library. Two backends implement the same interface:
  - `backend_cpu.js` — Float32Array ops, pure loops
  - `backend_webgpu.js` — GPUBuffer ops, WGSL shaders
- **Tape-based autograd** — forward ops append to a global tape array, `backward()` walks it in reverse. No topo sort needed.
- **ops.js** — each op computes forward via the backend, wraps result in a Tensor, and attaches a `_backwardFn` closure for backward.
- **train.js** — processes tokens sequentially (matching the Python version), uses KV cache for attention. Exports an async generator `train()` consumed by both Node CLI and browser.
- **bench.js** — finite-difference gradient checking + op-level timing.

## Key conventions

- Pure ES modules throughout (`import`/`export`), no bundler.
- No npm dependencies. Zero.
- Weights stored as `[outDim, inDim]` matching microgpt.py. The `matmulWT` op handles the transpose.
- `clearTape()` must be called at the start of each training step.
- The CPU backend is the reference — all GPU results are tested against it.

## Testing

```bash
node bench.js        # runs correctness tests + gradient checks + benchmarks
```

Correctness criteria:
- Forward: max absolute error < 1e-5 vs expected values
- Backward: gradient matches finite-difference (eps=1e-4, relative tolerance < 5%)
