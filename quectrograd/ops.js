// quectrograd forward ops with backward closures
// IMPORTANT: No op calls .toArray() — all computation uses backend handle methods.
// This keeps everything synchronous for GPU dispatch and data stays on device.

import { Tensor } from './tensor.js';
import { tape } from './autograd.js';

function record(out, op, parents, backwardFn, saved = {}) {
  out._op = op;
  out._parents = parents;
  out._backwardFn = backwardFn;
  out._saved = saved;
  tape.push(out);
  return out;
}

function needsGrad(t) {
  return t.requiresGrad || t._backwardFn;
}

// C = A @ B — A[M,K], B[K,N] → C[M,N]
export function matmul(a, b) {
  const M = a.shape[0], K = a.shape[1], N = b.shape[1];
  const handle = a.backend.matmul(a.handle, b.handle, M, K, N);
  const out = new Tensor(handle, [M, N], a.backend);
  return record(out, 'matmul', [a, b], () => {
    out.ensureGrad();
    const { dA, dB } = a.backend.matmulBackward(out.grad.handle, a.handle, b.handle, M, K, N);
    if (needsGrad(a)) { a.ensureGrad(); a.backend.accumulate(a.grad.handle, dA, M * K); }
    if (needsGrad(b)) { b.ensureGrad(); b.backend.accumulate(b.grad.handle, dB, K * N); }
  });
}

// out = X @ W^T — X[M,K], W[N,K] → out[M,N]  (matches Python's linear(x, w))
export function matmulWT(x, w) {
  const M = x.shape[0], K = x.shape[1], N = w.shape[0];
  const handle = x.backend.matmulWT(x.handle, w.handle, M, K, N);
  const out = new Tensor(handle, [M, N], x.backend);
  return record(out, 'matmulWT', [x, w], () => {
    out.ensureGrad();
    const { dx, dw } = x.backend.matmulWTBackward(out.grad.handle, x.handle, w.handle, M, K, N);
    if (needsGrad(x)) { x.ensureGrad(); x.backend.accumulate(x.grad.handle, dx, M * K); }
    if (needsGrad(w)) { w.ensureGrad(); w.backend.accumulate(w.grad.handle, dw, N * K); }
  });
}

// C = A + B (elementwise)
export function add(a, b) {
  const size = a.numel();
  const handle = a.backend.add(a.handle, b.handle, size);
  const out = new Tensor(handle, [...a.shape], a.backend);
  return record(out, 'add', [a, b], () => {
    out.ensureGrad();
    if (needsGrad(a)) { a.ensureGrad(); a.backend.accumulate(a.grad.handle, out.grad.handle, size); }
    if (needsGrad(b)) { b.ensureGrad(); b.backend.accumulate(b.grad.handle, out.grad.handle, size); }
  });
}

// y = relu(x)
export function relu(x) {
  const size = x.numel();
  const handle = x.backend.relu(x.handle, size);
  const out = new Tensor(handle, [...x.shape], x.backend);
  return record(out, 'relu', [x], () => {
    out.ensureGrad();
    if (needsGrad(x)) {
      x.ensureGrad();
      const dx = x.backend.reluBackward(out.grad.handle, x.handle, size);
      x.backend.accumulate(x.grad.handle, dx, size);
    }
  });
}

// y = log(x)
export function log(x) {
  const size = x.numel();
  const handle = x.backend.log(x.handle, size);
  const out = new Tensor(handle, [...x.shape], x.backend);
  return record(out, 'log', [x], () => {
    out.ensureGrad();
    if (needsGrad(x)) {
      x.ensureGrad();
      const dx = x.backend.logBackward(out.grad.handle, x.handle, size);
      x.backend.accumulate(x.grad.handle, dx, size);
    }
  });
}

// y = exp(x)
export function exp(x) {
  const size = x.numel();
  const handle = x.backend.exp(x.handle, size);
  const out = new Tensor(handle, [...x.shape], x.backend);
  return record(out, 'exp', [x], () => {
    out.ensureGrad();
    if (needsGrad(x)) {
      x.ensureGrad();
      const dx = x.backend.expBackward(out.grad.handle, out.handle, size);
      x.backend.accumulate(x.grad.handle, dx, size);
    }
  });
}

// y = x * scalar
export function scale(x, s) {
  const size = x.numel();
  const handle = x.backend.scale(x.handle, s, size);
  const out = new Tensor(handle, [...x.shape], x.backend);
  return record(out, 'scale', [x], () => {
    out.ensureGrad();
    if (needsGrad(x)) {
      x.ensureGrad();
      const dx = x.backend.scaleBackward(out.grad.handle, s, size);
      x.backend.accumulate(x.grad.handle, dx, size);
    }
  });
}

// y = -x
export function negate(x) {
  const size = x.numel();
  const handle = x.backend.negate(x.handle, size);
  const out = new Tensor(handle, [...x.shape], x.backend);
  return record(out, 'negate', [x], () => {
    out.ensureGrad();
    if (needsGrad(x)) {
      x.ensureGrad();
      const dx = x.backend.negateBackward(out.grad.handle, size);
      x.backend.accumulate(x.grad.handle, dx, size);
    }
  });
}

// Gather rows from embedding table
export function embedding(table, ids) {
  const vocabSize = table.shape[0], embdDim = table.shape[1];
  const seqLen = ids.length;
  const handle = table.backend.embedding(table.handle, ids, vocabSize, embdDim, seqLen);
  const out = new Tensor(handle, [seqLen, embdDim], table.backend);
  return record(out, 'embedding', [table], () => {
    out.ensureGrad();
    if (needsGrad(table)) {
      table.ensureGrad();
      const dTable = table.backend.embeddingBackward(out.grad.handle, ids, vocabSize, embdDim, seqLen);
      table.backend.accumulate(table.grad.handle, dTable, vocabSize * embdDim);
    }
  });
}

// RMS normalization per row
export function rmsnorm(x) {
  const rows = x.shape[0], cols = x.shape[1];
  const handle = x.backend.rmsnorm(x.handle, rows, cols);
  const out = new Tensor(handle, [rows, cols], x.backend);
  return record(out, 'rmsnorm', [x], () => {
    out.ensureGrad();
    if (needsGrad(x)) {
      x.ensureGrad();
      const dx = x.backend.rmsnormBackward(out.grad.handle, x.handle, rows, cols);
      x.backend.accumulate(x.grad.handle, dx, rows * cols);
    }
  });
}

// Softmax along last dim
export function softmax(x) {
  const rows = x.shape[0], cols = x.shape[1];
  const handle = x.backend.softmax(x.handle, rows, cols);
  const out = new Tensor(handle, [rows, cols], x.backend);
  return record(out, 'softmax', [x], () => {
    out.ensureGrad();
    if (needsGrad(x)) {
      x.ensureGrad();
      const dx = x.backend.softmaxBackward(out.grad.handle, out.handle, rows, cols);
      x.backend.accumulate(x.grad.handle, dx, rows * cols);
    }
  });
}

// Causal mask: set upper triangular (col > row) to -inf
export function causalMask(x) {
  const size = x.shape[0]; // square matrix [size, size]
  const handle = x.backend.causalMask(x.handle, size);
  const out = new Tensor(handle, [size, size], x.backend);
  return record(out, 'causalMask', [x], () => {
    out.ensureGrad();
    if (needsGrad(x)) {
      x.ensureGrad();
      const dx = x.backend.causalMaskBackward(out.grad.handle, size);
      x.backend.accumulate(x.grad.handle, dx, size * size);
    }
  });
}

// --- Data movement ops (differentiable) ---

// Slice columns: x[rows, totalCols] → out[rows, width] from colStart
export function sliceCols(x, colStart, width) {
  const rows = x.shape[0], totalCols = x.shape[1];
  const backend = x.backend;
  const handle = backend.sliceCols(x.handle, rows, totalCols, colStart, width);
  const result = new Tensor(handle, [rows, width], backend);
  return record(result, 'sliceCols', [x], () => {
    result.ensureGrad();
    if (needsGrad(x)) {
      x.ensureGrad();
      const dx = backend.sliceColsBackward(result.grad.handle, rows, totalCols, colStart, width);
      backend.accumulate(x.grad.handle, dx, rows * totalCols);
    }
  });
}

// Concatenate tensors along columns
export function concatCols(tensors) {
  const rows = tensors[0].shape[0];
  const backend = tensors[0].backend;
  const widths = tensors.map(t => t.shape[1]);
  const totalCols = widths.reduce((a, b) => a + b, 0);
  const handles = tensors.map(t => t.handle);
  const handle = backend.concatCols(handles, rows, widths);
  const result = new Tensor(handle, [rows, totalCols], backend);
  return record(result, 'concatCols', tensors, () => {
    result.ensureGrad();
    const grads = backend.splitCols(result.grad.handle, rows, widths);
    for (let i = 0; i < tensors.length; i++) {
      if (needsGrad(tensors[i])) {
        tensors[i].ensureGrad();
        backend.accumulate(tensors[i].grad.handle, grads[i], rows * widths[i]);
      }
    }
  });
}

// Stack [1, cols] tensors vertically → [n, cols]
export function stackRows(tensors) {
  const cols = tensors[0].shape[1];
  const n = tensors.length;
  const backend = tensors[0].backend;
  const handles = tensors.map(t => t.handle);
  const handle = backend.stackRows(handles, n, cols);
  const result = new Tensor(handle, [n, cols], backend);
  return record(result, 'stackRows', tensors, () => {
    result.ensureGrad();
    const grads = backend.splitRows(result.grad.handle, n, cols);
    for (let i = 0; i < n; i++) {
      if (needsGrad(tensors[i])) {
        tensors[i].ensureGrad();
        backend.accumulate(tensors[i].grad.handle, grads[i], cols);
      }
    }
  });
}

// Transpose 2D: [rows, cols] → [cols, rows]
export function transpose2D(x) {
  const rows = x.shape[0], cols = x.shape[1];
  const backend = x.backend;
  const handle = backend.transpose2D(x.handle, rows, cols);
  const result = new Tensor(handle, [cols, rows], backend);
  return record(result, 'transpose2D', [x], () => {
    result.ensureGrad();
    if (needsGrad(x)) {
      x.ensureGrad();
      // Backward of transpose is just transpose again (with swapped dims)
      const dx = backend.transpose2D(result.grad.handle, cols, rows);
      backend.accumulate(x.grad.handle, dx, rows * cols);
    }
  });
}

// Fused cross-entropy loss
export function crossEntropyLoss(logits, targets) {
  const rows = logits.shape[0], cols = logits.shape[1];
  const backend = logits.backend;
  const { lossHandle, softmaxOut } = backend.crossEntropyLoss(logits.handle, targets, rows, cols);
  const out = new Tensor(lossHandle, [1], backend);
  return record(out, 'crossEntropyLoss', [logits], () => {
    // Note: assumes this is the loss root (upstream grad = 1.0), which is always
    // the case in our training loop. For generality, would need to scale by out.grad.
    if (needsGrad(logits)) {
      logits.ensureGrad();
      const dLogits = backend.crossEntropyBackward(softmaxOut, targets, rows, cols);
      backend.accumulate(logits.grad.handle, dLogits, rows * cols);
    }
  });
}
