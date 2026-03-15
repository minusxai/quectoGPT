// quectrograd CPU backend — pure JS, no dependencies
// Every op works on Float32Array handles. This is the reference implementation.

export function initCPU() {
  return {
    name: 'cpu',

    alloc(size) {
      return new Float32Array(size);
    },

    free(_handle) {
      // no-op for JS arrays
    },

    fromArray(arr) {
      return new Float32Array(arr);
    },

    toArray(handle, _size) {
      return new Float32Array(handle); // copy
    },

    // --- Forward ops ---

    // C[i,j] = sum_k A[i,k] * B[k,j]  — A is MxK, B is KxN, C is MxN
    matmul(a, b, M, K, N) {
      const out = new Float32Array(M * N);
      for (let i = 0; i < M; i++) {
        for (let k = 0; k < K; k++) {
          const aik = a[i * K + k];
          for (let j = 0; j < N; j++) {
            out[i * N + j] += aik * b[k * N + j];
          }
        }
      }
      return out;
    },

    add(a, b, size) {
      const out = new Float32Array(size);
      for (let i = 0; i < size; i++) out[i] = a[i] + b[i];
      return out;
    },

    relu(x, size) {
      const out = new Float32Array(size);
      for (let i = 0; i < size; i++) out[i] = x[i] > 0 ? x[i] : 0;
      return out;
    },

    log(x, size) {
      const out = new Float32Array(size);
      for (let i = 0; i < size; i++) out[i] = Math.log(x[i]);
      return out;
    },

    exp(x, size) {
      const out = new Float32Array(size);
      for (let i = 0; i < size; i++) out[i] = Math.exp(x[i]);
      return out;
    },

    scale(x, s, size) {
      const out = new Float32Array(size);
      for (let i = 0; i < size; i++) out[i] = x[i] * s;
      return out;
    },

    negate(x, size) {
      const out = new Float32Array(size);
      for (let i = 0; i < size; i++) out[i] = -x[i];
      return out;
    },

    // Gather rows from embedding table. ids is Int32Array of length seqLen.
    // table is (vocabSize * embdDim), output is (seqLen * embdDim)
    embedding(table, ids, vocabSize, embdDim, seqLen) {
      const out = new Float32Array(seqLen * embdDim);
      for (let i = 0; i < seqLen; i++) {
        const row = ids[i];
        for (let j = 0; j < embdDim; j++) {
          out[i * embdDim + j] = table[row * embdDim + j];
        }
      }
      return out;
    },

    // RMS norm along last dimension. x is (rows * cols), normalizes each row.
    rmsnorm(x, rows, cols) {
      const out = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        let ms = 0;
        const off = r * cols;
        for (let c = 0; c < cols; c++) ms += x[off + c] * x[off + c];
        ms = ms / cols;
        const scale = 1.0 / Math.sqrt(ms + 1e-5);
        for (let c = 0; c < cols; c++) out[off + c] = x[off + c] * scale;
      }
      return out;
    },

    // Softmax along last dimension. x is (rows * cols)
    softmax(x, rows, cols) {
      const out = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        const off = r * cols;
        let max = -Infinity;
        for (let c = 0; c < cols; c++) if (x[off + c] > max) max = x[off + c];
        let sum = 0;
        for (let c = 0; c < cols; c++) {
          out[off + c] = Math.exp(x[off + c] - max);
          sum += out[off + c];
        }
        for (let c = 0; c < cols; c++) out[off + c] /= sum;
      }
      return out;
    },

    // Cross-entropy loss: -log(softmax(logits)[target]) for each row
    // logits is (rows * cols), targets is Int32Array of length rows
    // Returns { loss: Float32Array(rows), softmaxOut: Float32Array(rows * cols) }
    crossEntropyLoss(logits, targets, rows, cols) {
      const probs = this.softmax(logits, rows, cols);
      const loss = new Float32Array(rows);
      for (let r = 0; r < rows; r++) {
        const p = probs[r * cols + targets[r]];
        loss[r] = -Math.log(Math.max(p, 1e-10));
      }
      return { loss, softmaxOut: probs };
    },

    // --- Backward ops ---

    // dA = dC @ B^T, dB = A^T @ dC
    matmulBackward(dC, a, b, M, K, N) {
      // dA[M,K] = dC[M,N] @ B^T[N,K]
      const dA = new Float32Array(M * K);
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
          const dc = dC[i * N + j];
          for (let k = 0; k < K; k++) {
            dA[i * K + k] += dc * b[k * N + j];
          }
        }
      }
      // dB[K,N] = A^T[K,M] @ dC[M,N]
      const dB = new Float32Array(K * N);
      for (let k = 0; k < K; k++) {
        for (let i = 0; i < M; i++) {
          const aik = a[i * K + k];
          for (let j = 0; j < N; j++) {
            dB[k * N + j] += aik * dC[i * N + j];
          }
        }
      }
      return { dA, dB };
    },

    addBackward(dC, size) {
      // dA = dC, dB = dC (copy)
      return { dA: new Float32Array(dC), dB: new Float32Array(dC) };
    },

    reluBackward(dOut, x, size) {
      const dx = new Float32Array(size);
      for (let i = 0; i < size; i++) dx[i] = x[i] > 0 ? dOut[i] : 0;
      return dx;
    },

    logBackward(dOut, x, size) {
      const dx = new Float32Array(size);
      for (let i = 0; i < size; i++) dx[i] = dOut[i] / x[i];
      return dx;
    },

    expBackward(dOut, expOut, size) {
      const dx = new Float32Array(size);
      for (let i = 0; i < size; i++) dx[i] = dOut[i] * expOut[i];
      return dx;
    },

    scaleBackward(dOut, s, size) {
      const dx = new Float32Array(size);
      for (let i = 0; i < size; i++) dx[i] = dOut[i] * s;
      return dx;
    },

    negateBackward(dOut, size) {
      const dx = new Float32Array(size);
      for (let i = 0; i < size; i++) dx[i] = -dOut[i];
      return dx;
    },

    // Scatter-add gradient back to embedding table rows
    embeddingBackward(dOut, ids, vocabSize, embdDim, seqLen) {
      const dTable = new Float32Array(vocabSize * embdDim);
      for (let i = 0; i < seqLen; i++) {
        const row = ids[i];
        for (let j = 0; j < embdDim; j++) {
          dTable[row * embdDim + j] += dOut[i * embdDim + j];
        }
      }
      return dTable;
    },

    // RMS norm backward
    rmsnormBackward(dOut, x, rows, cols) {
      const dx = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        const off = r * cols;
        let ms = 0;
        for (let c = 0; c < cols; c++) ms += x[off + c] * x[off + c];
        ms = ms / cols;
        const invRms = 1.0 / Math.sqrt(ms + 1e-5);
        // dot(dOut, x) for this row
        let dotDX = 0;
        for (let c = 0; c < cols; c++) dotDX += dOut[off + c] * x[off + c];
        for (let c = 0; c < cols; c++) {
          dx[off + c] = invRms * (dOut[off + c] - x[off + c] * dotDX * invRms * invRms / cols);
        }
      }
      return dx;
    },

    // Softmax backward: dx = softmaxOut * (dOut - sum(dOut * softmaxOut))
    softmaxBackward(dOut, softmaxOut, rows, cols) {
      const dx = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        const off = r * cols;
        let dot = 0;
        for (let c = 0; c < cols; c++) dot += dOut[off + c] * softmaxOut[off + c];
        for (let c = 0; c < cols; c++) {
          dx[off + c] = softmaxOut[off + c] * (dOut[off + c] - dot);
        }
      }
      return dx;
    },

    // Cross-entropy backward: dLogits = softmaxOut - oneHot(targets)
    // Divided by rows for mean loss
    crossEntropyBackward(softmaxOut, targets, rows, cols) {
      const dLogits = new Float32Array(softmaxOut);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          dLogits[r * cols + c] /= rows;
        }
        dLogits[r * cols + targets[r]] -= 1.0 / rows;
      }
      return dLogits;
    },

    // --- Optimizer ---
    adamUpdate(params, grads, mBuf, vBuf, lr, beta1, beta2, eps, step, count) {
      const bc1 = 1 - Math.pow(beta1, step);
      const bc2 = 1 - Math.pow(beta2, step);
      for (let i = 0; i < count; i++) {
        mBuf[i] = beta1 * mBuf[i] + (1 - beta1) * grads[i];
        vBuf[i] = beta2 * vBuf[i] + (1 - beta2) * grads[i] * grads[i];
        const mHat = mBuf[i] / bc1;
        const vHat = vBuf[i] / bc2;
        params[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    },

    // Utility: fill buffer with zeros
    zeros(size) {
      return new Float32Array(size);
    },

    // Utility: fill buffer with values from a normal distribution
    randn(size, std) {
      const out = new Float32Array(size);
      for (let i = 0; i < size; i += 2) {
        // Box-Muller transform
        const u1 = Math.random();
        const u2 = Math.random();
        const r = Math.sqrt(-2 * Math.log(u1));
        const theta = 2 * Math.PI * u2;
        out[i] = r * Math.cos(theta) * std;
        if (i + 1 < size) out[i + 1] = r * Math.sin(theta) * std;
      }
      return out;
    },

    // Accumulate: dst += src
    accumulate(dst, src, size) {
      for (let i = 0; i < size; i++) dst[i] += src[i];
    },

    // Fill with scalar
    fill(handle, value, size) {
      for (let i = 0; i < size; i++) handle[i] = value;
    },
  };
}
