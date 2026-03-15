// quectrograd forward ops with backward closures
// Each op: takes Tensor inputs → computes forward via backend → wraps in new Tensor → attaches _backwardFn

import { Tensor } from './tensor.js';
import { tape } from './autograd.js';

// Helper: record op on tape
function record(out, op, parents, backwardFn, saved = {}) {
  out._op = op;
  out._parents = parents;
  out._backwardFn = backwardFn;
  out._saved = saved;
  tape.push(out);
  return out;
}

// C = A @ B — A[M,K], B[K,N] → C[M,N]
export function matmul(a, b) {
  const M = a.shape[0], K = a.shape[1], N = b.shape[1];
  const handle = a.backend.matmul(a.handle, b.handle, M, K, N);
  const out = new Tensor(handle, [M, N], a.backend);
  return record(out, 'matmul', [a, b], () => {
    out.ensureGrad();
    const { dA, dB } = a.backend.matmulBackward(out.grad.handle, a.handle, b.handle, M, K, N);
    if (a.requiresGrad || a._backwardFn) {
      a.ensureGrad();
      a.backend.accumulate(a.grad.handle, dA, M * K);
    }
    if (b.requiresGrad || b._backwardFn) {
      b.ensureGrad();
      b.backend.accumulate(b.grad.handle, dB, K * N);
    }
  });
}

// C = A + B (elementwise, same shape)
export function add(a, b) {
  const size = a.numel();
  const handle = a.backend.add(a.handle, b.handle, size);
  const out = new Tensor(handle, [...a.shape], a.backend);
  return record(out, 'add', [a, b], () => {
    out.ensureGrad();
    if (a.requiresGrad || a._backwardFn) {
      a.ensureGrad();
      a.backend.accumulate(a.grad.handle, out.grad.handle, size);
    }
    if (b.requiresGrad || b._backwardFn) {
      b.ensureGrad();
      b.backend.accumulate(b.grad.handle, out.grad.handle, size);
    }
  });
}

// y = relu(x)
export function relu(x) {
  const size = x.numel();
  const handle = x.backend.relu(x.handle, size);
  const out = new Tensor(handle, [...x.shape], x.backend);
  return record(out, 'relu', [x], () => {
    out.ensureGrad();
    if (x.requiresGrad || x._backwardFn) {
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
    if (x.requiresGrad || x._backwardFn) {
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
    if (x.requiresGrad || x._backwardFn) {
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
    if (x.requiresGrad || x._backwardFn) {
      x.ensureGrad();
      const dx = x.backend.scaleBackward(out.grad.handle, s, size);
      x.backend.accumulate(x.grad.handle, dx, size);
    }
  }, { s });
}

// y = -x
export function negate(x) {
  const size = x.numel();
  const handle = x.backend.negate(x.handle, size);
  const out = new Tensor(handle, [...x.shape], x.backend);
  return record(out, 'negate', [x], () => {
    out.ensureGrad();
    if (x.requiresGrad || x._backwardFn) {
      x.ensureGrad();
      const dx = x.backend.negateBackward(out.grad.handle, size);
      x.backend.accumulate(x.grad.handle, dx, size);
    }
  });
}

// Gather rows from embedding table
// table: Tensor[vocabSize, embdDim], ids: Int32Array → out: Tensor[seqLen, embdDim]
export function embedding(table, ids) {
  const vocabSize = table.shape[0], embdDim = table.shape[1];
  const seqLen = ids.length;
  const handle = table.backend.embedding(table.handle, ids, vocabSize, embdDim, seqLen);
  const out = new Tensor(handle, [seqLen, embdDim], table.backend);
  return record(out, 'embedding', [table], () => {
    out.ensureGrad();
    if (table.requiresGrad || table._backwardFn) {
      table.ensureGrad();
      const dTable = table.backend.embeddingBackward(out.grad.handle, ids, vocabSize, embdDim, seqLen);
      table.backend.accumulate(table.grad.handle, dTable, vocabSize * embdDim);
    }
  }, { ids });
}

// RMS normalization: x[rows, cols] → normalized per row
export function rmsnorm(x) {
  const rows = x.shape[0], cols = x.shape[1];
  const handle = x.backend.rmsnorm(x.handle, rows, cols);
  const out = new Tensor(handle, [rows, cols], x.backend);
  return record(out, 'rmsnorm', [x], () => {
    out.ensureGrad();
    if (x.requiresGrad || x._backwardFn) {
      x.ensureGrad();
      const dx = x.backend.rmsnormBackward(out.grad.handle, x.handle, rows, cols);
      x.backend.accumulate(x.grad.handle, dx, rows * cols);
    }
  });
}

// Softmax along last dim: x[rows, cols] → probabilities per row
export function softmax(x) {
  const rows = x.shape[0], cols = x.shape[1];
  const handle = x.backend.softmax(x.handle, rows, cols);
  const out = new Tensor(handle, [rows, cols], x.backend);
  return record(out, 'softmax', [x], () => {
    out.ensureGrad();
    if (x.requiresGrad || x._backwardFn) {
      x.ensureGrad();
      const dx = x.backend.softmaxBackward(out.grad.handle, out.handle, rows, cols);
      x.backend.accumulate(x.grad.handle, dx, rows * cols);
    }
  });
}

// X[M,K] @ W^T[K,N] where W is [N,K] — matches Python's linear(x, w)
// Fused op: avoids materializing W^T. Backward flows to both x and w.
export function matmulWT(x, w) {
  const M = x.shape[0], K = x.shape[1];
  const N = w.shape[0]; // w is [N, K], output is [M, N]
  const backend = x.backend;

  // Forward: out[i,j] = sum_k x[i,k] * w[j,k]
  const xArr = x.toArray();
  const wArr = w.toArray();
  const outArr = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += xArr[i * K + k] * wArr[j * K + k];
      }
      outArr[i * N + j] = sum;
    }
  }

  const out = Tensor.from(outArr, [M, N], backend);
  return record(out, 'matmulWT', [x, w], () => {
    out.ensureGrad();
    const dOut = out.grad.toArray();

    if (x.requiresGrad || x._backwardFn) {
      x.ensureGrad();
      // dx[i,k] = sum_j dOut[i,j] * w[j,k]
      const dx = new Float32Array(M * K);
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
          const d = dOut[i * N + j];
          for (let k = 0; k < K; k++) {
            dx[i * K + k] += d * wArr[j * K + k];
          }
        }
      }
      backend.accumulate(x.grad.handle, dx, M * K);
    }

    if (w.requiresGrad || w._backwardFn) {
      w.ensureGrad();
      // dw[j,k] = sum_i dOut[i,j] * x[i,k]
      const dw = new Float32Array(N * K);
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
          const d = dOut[i * N + j];
          for (let k = 0; k < K; k++) {
            dw[j * K + k] += d * xArr[i * K + k];
          }
        }
      }
      backend.accumulate(w.grad.handle, dw, N * K);
    }
  });
}

// Slice columns: x[rows, totalCols] → out[rows, width] starting at colStart
export function sliceCols(x, colStart, width) {
  const rows = x.shape[0], totalCols = x.shape[1];
  const backend = x.backend;
  const arr = x.toArray();
  const out = new Float32Array(rows * width);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < width; c++) {
      out[r * width + c] = arr[r * totalCols + colStart + c];
    }
  }
  const result = Tensor.from(out, [rows, width], backend);
  return record(result, 'sliceCols', [x], () => {
    result.ensureGrad();
    if (x.requiresGrad || x._backwardFn) {
      x.ensureGrad();
      const dOut = result.grad.toArray();
      const dx = new Float32Array(rows * totalCols);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < width; c++) {
          dx[r * totalCols + colStart + c] = dOut[r * width + c];
        }
      }
      backend.accumulate(x.grad.handle, dx, rows * totalCols);
    }
  }, { colStart, width });
}

// Concatenate tensors along columns: [t1[rows,c1], t2[rows,c2], ...] → [rows, c1+c2+...]
export function concatCols(tensors) {
  const rows = tensors[0].shape[0];
  const backend = tensors[0].backend;
  const widths = tensors.map(t => t.shape[1]);
  const totalCols = widths.reduce((a, b) => a + b, 0);
  const out = new Float32Array(rows * totalCols);
  let colOff = 0;
  for (let ti = 0; ti < tensors.length; ti++) {
    const arr = tensors[ti].toArray();
    const w = widths[ti];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < w; c++) {
        out[r * totalCols + colOff + c] = arr[r * w + c];
      }
    }
    colOff += w;
  }
  const result = Tensor.from(out, [rows, totalCols], backend);
  return record(result, 'concatCols', tensors, () => {
    result.ensureGrad();
    const dOut = result.grad.toArray();
    let off = 0;
    for (let ti = 0; ti < tensors.length; ti++) {
      const t = tensors[ti];
      if (t.requiresGrad || t._backwardFn) {
        t.ensureGrad();
        const w = widths[ti];
        const dt = new Float32Array(rows * w);
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < w; c++) {
            dt[r * w + c] = dOut[r * totalCols + off + c];
          }
        }
        backend.accumulate(t.grad.handle, dt, rows * w);
      }
      off += widths[ti];
    }
  });
}

// Stack [1, cols] tensors vertically into [n, cols]
export function stackRows(tensors) {
  const cols = tensors[0].shape[1];
  const n = tensors.length;
  const backend = tensors[0].backend;
  const out = new Float32Array(n * cols);
  for (let i = 0; i < n; i++) {
    out.set(tensors[i].toArray(), i * cols);
  }
  const result = Tensor.from(out, [n, cols], backend);
  return record(result, 'stackRows', tensors, () => {
    result.ensureGrad();
    const dOut = result.grad.toArray();
    for (let i = 0; i < n; i++) {
      const t = tensors[i];
      if (t.requiresGrad || t._backwardFn) {
        t.ensureGrad();
        const dt = dOut.slice(i * cols, (i + 1) * cols);
        backend.accumulate(t.grad.handle, dt, cols);
      }
    }
  });
}

// Transpose 2D: [rows, cols] → [cols, rows]
export function transpose2D(x) {
  const rows = x.shape[0], cols = x.shape[1];
  const backend = x.backend;
  const arr = x.toArray();
  const out = new Float32Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[c * rows + r] = arr[r * cols + c];
    }
  }
  const result = Tensor.from(out, [cols, rows], backend);
  return record(result, 'transpose2D', [x], () => {
    result.ensureGrad();
    if (x.requiresGrad || x._backwardFn) {
      x.ensureGrad();
      const dOut = result.grad.toArray();
      const dx = new Float32Array(rows * cols);
      for (let r = 0; r < cols; r++) {
        for (let c = 0; c < rows; c++) {
          dx[c * cols + r] = dOut[r * rows + c];
        }
      }
      backend.accumulate(x.grad.handle, dx, rows * cols);
    }
  });
}

// Fused cross-entropy loss: logits[rows, cols], targets: Int32Array[rows] → scalar loss
// Returns { loss: Tensor[1], softmaxOut: saved for backward }
export function crossEntropyLoss(logits, targets) {
  const rows = logits.shape[0], cols = logits.shape[1];
  const { loss: lossArr, softmaxOut } = logits.backend.crossEntropyLoss(logits.handle, targets, rows, cols);
  // Mean loss
  let mean = 0;
  for (let i = 0; i < rows; i++) mean += lossArr[i];
  mean /= rows;
  const lossHandle = logits.backend.fromArray(new Float32Array([mean]));
  const out = new Tensor(lossHandle, [1], logits.backend);
  return record(out, 'crossEntropyLoss', [logits], () => {
    out.ensureGrad();
    if (logits.requiresGrad || logits._backwardFn) {
      logits.ensureGrad();
      const dLogits = logits.backend.crossEntropyBackward(softmaxOut, targets, rows, cols);
      // Scale by upstream gradient
      const outGrad = out.grad.toArray()[0];
      if (outGrad !== 1.0) {
        for (let i = 0; i < dLogits.length; i++) dLogits[i] *= outGrad;
      }
      logits.backend.accumulate(logits.grad.handle, dLogits, rows * cols);
    }
  }, { softmaxOut, targets });
}
