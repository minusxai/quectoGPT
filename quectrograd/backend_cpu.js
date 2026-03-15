// quectrograd CPU backend — pure JS, no dependencies
// Every op works on Float32Array handles. This is the reference implementation.

export function initCPU() {
  return {
    name: 'cpu',

    alloc(size) {
      return new Float32Array(size);
    },

    free(_handle) {},

    fromArray(arr) {
      return new Float32Array(arr);
    },

    toArray(handle, _size) {
      return new Float32Array(handle);
    },

    // --- Forward ops ---

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

    // X[M,K] @ W^T => out[M,N], where W is [N,K]
    // out[i,j] = sum_k x[i,k] * w[j,k]
    matmulWT(x, w, M, K, N) {
      const out = new Float32Array(M * N);
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
          let sum = 0;
          for (let k = 0; k < K; k++) sum += x[i * K + k] * w[j * K + k];
          out[i * N + j] = sum;
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

    rmsnorm(x, rows, cols) {
      const out = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        let ms = 0;
        const off = r * cols;
        for (let c = 0; c < cols; c++) ms += x[off + c] * x[off + c];
        ms = ms / cols;
        const s = 1.0 / Math.sqrt(ms + 1e-5);
        for (let c = 0; c < cols; c++) out[off + c] = x[off + c] * s;
      }
      return out;
    },

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

    // Fused cross-entropy: softmax + -log(p[target]), returns mean scalar + saved softmax
    // Returns { lossHandle: Float32Array(1), softmaxOut: Float32Array }
    crossEntropyLoss(logits, targets, rows, cols) {
      const probs = this.softmax(logits, rows, cols);
      let meanLoss = 0;
      for (let r = 0; r < rows; r++) {
        meanLoss += -Math.log(Math.max(probs[r * cols + targets[r]], 1e-10));
      }
      meanLoss /= rows;
      return { lossHandle: new Float32Array([meanLoss]), softmaxOut: probs };
    },

    // --- Data movement ops ---

    // Slice columns: x[rows, totalCols] -> out[rows, width] starting at colStart
    sliceCols(x, rows, totalCols, colStart, width) {
      const out = new Float32Array(rows * width);
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < width; c++)
          out[r * width + c] = x[r * totalCols + colStart + c];
      return out;
    },

    // Concat columns: multiple [rows, w_i] handles -> [rows, sum(w_i)]
    concatCols(handles, rows, widths) {
      const totalCols = widths.reduce((a, b) => a + b, 0);
      const out = new Float32Array(rows * totalCols);
      let colOff = 0;
      for (let ti = 0; ti < handles.length; ti++) {
        const w = widths[ti];
        for (let r = 0; r < rows; r++)
          for (let c = 0; c < w; c++)
            out[r * totalCols + colOff + c] = handles[ti][r * w + c];
        colOff += w;
      }
      return out;
    },

    // Stack rows: n handles of [1, cols] -> [n, cols]
    stackRows(handles, n, cols) {
      const out = new Float32Array(n * cols);
      for (let i = 0; i < n; i++) out.set(handles[i], i * cols);
      return out;
    },

    // Transpose 2D: [rows, cols] -> [cols, rows]
    transpose2D(x, rows, cols) {
      const out = new Float32Array(cols * rows);
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++)
          out[c * rows + r] = x[r * cols + c];
      return out;
    },

    // --- Backward ops ---

    matmulBackward(dC, a, b, M, K, N) {
      const dA = new Float32Array(M * K);
      for (let i = 0; i < M; i++)
        for (let j = 0; j < N; j++) {
          const dc = dC[i * N + j];
          for (let k = 0; k < K; k++) dA[i * K + k] += dc * b[k * N + j];
        }
      const dB = new Float32Array(K * N);
      for (let k = 0; k < K; k++)
        for (let i = 0; i < M; i++) {
          const aik = a[i * K + k];
          for (let j = 0; j < N; j++) dB[k * N + j] += aik * dC[i * N + j];
        }
      return { dA, dB };
    },

    // matmulWT backward: dx = dOut @ W (regular matmul), dw = dOut^T @ X
    matmulWTBackward(dOut, x, w, M, K, N) {
      // dx[i,k] = sum_j dOut[i,j] * w[j,k]  =>  dx = matmul(dOut[M,N], W[N,K])
      const dx = this.matmul(dOut, w, M, N, K);
      // dw[j,k] = sum_i dOut[i,j] * x[i,k]  =>  dw = matmul(dOut^T[N,M], X[M,K])
      const dOutT = this.transpose2D(dOut, M, N);
      const dw = this.matmul(dOutT, x, N, M, K);
      return { dx, dw };
    },

    addBackward(dC, size) {
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

    embeddingBackward(dOut, ids, vocabSize, embdDim, seqLen) {
      const dTable = new Float32Array(vocabSize * embdDim);
      for (let i = 0; i < seqLen; i++) {
        const row = ids[i];
        for (let j = 0; j < embdDim; j++) dTable[row * embdDim + j] += dOut[i * embdDim + j];
      }
      return dTable;
    },

    rmsnormBackward(dOut, x, rows, cols) {
      const dx = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        const off = r * cols;
        let ms = 0;
        for (let c = 0; c < cols; c++) ms += x[off + c] * x[off + c];
        ms = ms / cols;
        const invRms = 1.0 / Math.sqrt(ms + 1e-5);
        let dotDX = 0;
        for (let c = 0; c < cols; c++) dotDX += dOut[off + c] * x[off + c];
        for (let c = 0; c < cols; c++) {
          dx[off + c] = invRms * (dOut[off + c] - x[off + c] * dotDX * invRms * invRms / cols);
        }
      }
      return dx;
    },

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

    crossEntropyBackward(softmaxOut, targets, rows, cols) {
      const dLogits = new Float32Array(softmaxOut);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) dLogits[r * cols + c] /= rows;
        dLogits[r * cols + targets[r]] -= 1.0 / rows;
      }
      return dLogits;
    },

    // Backward for sliceCols: scatter dOut[rows, width] into zeros[rows, totalCols] at colStart
    sliceColsBackward(dOut, rows, totalCols, colStart, width) {
      const dx = new Float32Array(rows * totalCols);
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < width; c++)
          dx[r * totalCols + colStart + c] = dOut[r * width + c];
      return dx;
    },

    // Backward for concatCols: split dOut[rows, totalCols] into per-input gradients
    splitCols(dOut, rows, widths) {
      const totalCols = widths.reduce((a, b) => a + b, 0);
      const results = [];
      let colOff = 0;
      for (const w of widths) {
        const out = new Float32Array(rows * w);
        for (let r = 0; r < rows; r++)
          for (let c = 0; c < w; c++)
            out[r * w + c] = dOut[r * totalCols + colOff + c];
        results.push(out);
        colOff += w;
      }
      return results;
    },

    // Backward for stackRows: split dOut[n, cols] into n handles
    splitRows(handle, n, cols) {
      const results = [];
      for (let i = 0; i < n; i++)
        results.push(handle.slice(i * cols, (i + 1) * cols));
      return results;
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

    // --- Utilities ---
    zeros(size) {
      return new Float32Array(size);
    },

    randn(size, std) {
      const out = new Float32Array(size);
      for (let i = 0; i < size; i += 2) {
        const u1 = Math.random();
        const u2 = Math.random();
        const r = Math.sqrt(-2 * Math.log(u1));
        const theta = 2 * Math.PI * u2;
        out[i] = r * Math.cos(theta) * std;
        if (i + 1 < size) out[i + 1] = r * Math.sin(theta) * std;
      }
      return out;
    },

    accumulate(dst, src, size) {
      for (let i = 0; i < size; i++) dst[i] += src[i];
    },

    fill(handle, value, size) {
      for (let i = 0; i < size; i++) handle[i] = value;
    },
  };
}
