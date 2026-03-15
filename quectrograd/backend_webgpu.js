// quectrograd WebGPU backend — WGSL compute shaders
// All ops work on GPUBuffer handles. toArray() is async (GPU readback).
// No op in ops.js calls toArray — data stays on GPU throughout forward/backward.

export async function initWebGPU() {
  if (!navigator.gpu) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter found');
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    }
  });

  const pipelineCache = new Map();

  function getOrCreatePipeline(key, code, entryPoint = 'main') {
    if (pipelineCache.has(key)) return pipelineCache.get(key);
    const module = device.createShaderModule({ code });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint },
    });
    pipelineCache.set(key, pipeline);
    return pipeline;
  }

  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  function createGPUBuffer(data) {
    const buf = device.createBuffer({ size: data.byteLength, usage: STORAGE, mappedAtCreation: true });
    new Float32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }

  function emptyBuf(floatCount) {
    return device.createBuffer({ size: Math.max(floatCount * 4, 4), usage: STORAGE });
  }

  function uniformBuf(uint32Values) {
    const data = new Uint32Array(uint32Values);
    const size = Math.max(data.byteLength, 16); // min 16 bytes for uniform
    const buf = device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }

  // Pack float as uint32 bit pattern for uniform buffers
  function f2u(f) {
    const a = new Float32Array([f]);
    return new Uint32Array(a.buffer)[0];
  }

  function dispatch(pipeline, buffers, wgX, wgY = 1, wgZ = 1) {
    const entries = buffers.map((buf, i) => ({ binding: i, resource: { buffer: buf } }));
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY, wgZ);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  const wg = (n) => Math.ceil(n / 256);

  // ===== WGSL Shaders =====

  const unary = (op) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= arrayLength(&out)) { return; } ${op}
}`;

  const binary = (op) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= arrayLength(&out)) { return; } ${op}
}`;

  const matmulWGSL = `
struct P { M: u32, K: u32, N: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x; let c = gid.y;
  if (r >= p.M || c >= p.N) { return; }
  var s: f32 = 0.0;
  for (var k: u32 = 0; k < p.K; k++) { s += a[r * p.K + k] * b[k * p.N + c]; }
  out[r * p.N + c] = s;
}`;

  // X[M,K] @ W[N,K]^T = out[M,N], where out[i,j] = sum_k x[i,k] * w[j,k]
  const matmulWTWGSL = `
struct P { M: u32, K: u32, N: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x; let c = gid.y;
  if (r >= p.M || c >= p.N) { return; }
  var s: f32 = 0.0;
  for (var k: u32 = 0; k < p.K; k++) { s += x[r * p.K + k] * w[c * p.K + k]; }
  out[r * p.N + c] = s;
}`;

  const softmaxWGSL = `
struct P { rows: u32, cols: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x; if (r >= p.rows) { return; }
  let off = r * p.cols;
  var mx: f32 = x[off];
  for (var c: u32 = 1; c < p.cols; c++) { mx = max(mx, x[off + c]); }
  var sm: f32 = 0.0;
  for (var c: u32 = 0; c < p.cols; c++) { out[off + c] = exp(x[off + c] - mx); sm += out[off + c]; }
  for (var c: u32 = 0; c < p.cols; c++) { out[off + c] /= sm; }
}`;

  const rmsnormWGSL = `
struct P { rows: u32, cols: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x; if (r >= p.rows) { return; }
  let off = r * p.cols;
  var ms: f32 = 0.0;
  for (var c: u32 = 0; c < p.cols; c++) { ms += x[off + c] * x[off + c]; }
  ms /= f32(p.cols);
  let s = 1.0 / sqrt(ms + 1e-5);
  for (var c: u32 = 0; c < p.cols; c++) { out[off + c] = x[off + c] * s; }
}`;

  const embeddingWGSL = `
struct P { dim: u32, seq: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x; if (idx >= p.seq * p.dim) { return; }
  let i = idx / p.dim; let j = idx % p.dim;
  out[i * p.dim + j] = table[ids[i] * p.dim + j];
}`;

  const transpose2DWGSL = `
struct P { rows: u32, cols: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x; if (idx >= p.rows * p.cols) { return; }
  let r = idx / p.cols; let c = idx % p.cols;
  dst[c * p.rows + r] = src[r * p.cols + c];
}`;

  const sliceColsWGSL = `
struct P { rows: u32, srcCols: u32, start: u32, width: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x; if (idx >= p.rows * p.width) { return; }
  let r = idx / p.width; let c = idx % p.width;
  dst[r * p.width + c] = src[r * p.srcCols + p.start + c];
}`;

  // Backward of sliceCols: scatter into zeroed buffer at column offset
  const sliceColsBackwardWGSL = `
struct P { rows: u32, dstCols: u32, start: u32, width: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x; if (idx >= p.rows * p.width) { return; }
  let r = idx / p.width; let c = idx % p.width;
  dst[r * p.dstCols + p.start + c] = src[r * p.width + c];
}`;

  // Copy one input's columns into a destination buffer at a column offset
  const copyColsWGSL = `
struct P { rows: u32, srcCols: u32, dstCols: u32, dstOff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x; if (idx >= p.rows * p.srcCols) { return; }
  let r = idx / p.srcCols; let c = idx % p.srcCols;
  dst[r * p.dstCols + p.dstOff + c] = src[r * p.srcCols + c];
}`;

  // Extract columns from src at offset (backward of concatCols = splitCols)
  const extractColsWGSL = `
struct P { rows: u32, srcCols: u32, dstCols: u32, srcOff: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x; if (idx >= p.rows * p.dstCols) { return; }
  let r = idx / p.dstCols; let c = idx % p.dstCols;
  dst[r * p.dstCols + c] = src[r * p.srcCols + p.srcOff + c];
}`;

  // Fused cross-entropy: softmax + mean(-log(p[target]))
  // Single workgroup — fine for small sequence lengths
  const crossEntropyWGSL = `
struct P { rows: u32, cols: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> probs: array<f32>;
@group(0) @binding(4) var<storage, read_write> loss: array<f32>;
@compute @workgroup_size(1) fn main() {
  var total: f32 = 0.0;
  for (var r: u32 = 0; r < p.rows; r++) {
    let off = r * p.cols;
    var mx: f32 = logits[off];
    for (var c: u32 = 1; c < p.cols; c++) { mx = max(mx, logits[off + c]); }
    var sm: f32 = 0.0;
    for (var c: u32 = 0; c < p.cols; c++) { probs[off + c] = exp(logits[off + c] - mx); sm += probs[off + c]; }
    for (var c: u32 = 0; c < p.cols; c++) { probs[off + c] /= sm; }
    total += -log(max(probs[off + targets[r]], 1e-10));
  }
  loss[0] = total / f32(p.rows);
}`;

  const crossEntropyBackwardWGSL = `
struct P { rows: u32, cols: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> dlogits: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x; if (idx >= p.rows * p.cols) { return; }
  let r = idx / p.cols; let c = idx % p.cols;
  var val = probs[idx] / f32(p.rows);
  if (c == targets[r]) { val -= 1.0 / f32(p.rows); }
  dlogits[idx] = val;
}`;

  // Backward shaders for existing ops
  const reluBackwardWGSL = `
@group(0) @binding(0) var<storage, read> dout: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= arrayLength(&dx)) { return; }
  dx[i] = select(0.0, dout[i], x[i] > 0.0);
}`;

  const softmaxBackwardWGSL = `
struct P { rows: u32, cols: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> sout: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x; if (r >= p.rows) { return; }
  let off = r * p.cols;
  var d: f32 = 0.0;
  for (var c: u32 = 0; c < p.cols; c++) { d += dout[off + c] * sout[off + c]; }
  for (var c: u32 = 0; c < p.cols; c++) { dx[off + c] = sout[off + c] * (dout[off + c] - d); }
}`;

  const rmsnormBackwardWGSL = `
struct P { rows: u32, cols: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x; if (r >= p.rows) { return; }
  let off = r * p.cols;
  var ms: f32 = 0.0;
  for (var c: u32 = 0; c < p.cols; c++) { ms += x[off + c] * x[off + c]; }
  ms /= f32(p.cols);
  let inv = 1.0 / sqrt(ms + 1e-5);
  var dot: f32 = 0.0;
  for (var c: u32 = 0; c < p.cols; c++) { dot += dout[off + c] * x[off + c]; }
  for (var c: u32 = 0; c < p.cols; c++) {
    dx[off + c] = inv * (dout[off + c] - x[off + c] * dot * inv * inv / f32(p.cols));
  }
}`;

  const embeddingBackwardWGSL = `
struct P { vocab: u32, dim: u32, seq: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> dtable: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x; if (idx >= p.seq * p.dim) { return; }
  let i = idx / p.dim; let j = idx % p.dim;
  // Race condition for duplicate ids — acceptable for small vocab
  dtable[ids[i] * p.dim + j] += dout[idx];
}`;

  const accumulateWGSL = `
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= arrayLength(&dst)) { return; } dst[i] += src[i];
}`;

  const fillWGSL = `
struct P { val: f32, size: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= p.size) { return; } buf[i] = p.val;
}`;

  const adamWGSL = `
struct P { lr: f32, b1: f32, b2: f32, eps: f32, bc1: f32, bc2: f32, cnt: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= p.cnt) { return; }
  let g = grad[i];
  m[i] = p.b1 * m[i] + (1.0 - p.b1) * g;
  v[i] = p.b2 * v[i] + (1.0 - p.b2) * g * g;
  param[i] -= p.lr * (m[i] / p.bc1) / (sqrt(v[i] / p.bc2) + p.eps);
}`;

  // Helper: upload Int32Array as Uint32 storage buffer
  function uploadIds(ids) {
    const buf = device.createBuffer({ size: ids.length * 4, usage: STORAGE, mappedAtCreation: true });
    const u = new Uint32Array(buf.getMappedRange());
    if (ids instanceof Int32Array) {
      for (let i = 0; i < ids.length; i++) u[i] = ids[i] >>> 0;
    } else {
      u.set(ids);
    }
    buf.unmap();
    return buf;
  }

  // ===== Backend interface =====

  return {
    name: 'webgpu',
    device,

    alloc(size) { return emptyBuf(size); },
    free(h) { if (h?.destroy) h.destroy(); },
    fromArray(arr) { return createGPUBuffer(arr instanceof Float32Array ? arr : new Float32Array(arr)); },

    async toArray(handle, size) {
      const staging = device.createBuffer({ size: size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(handle, 0, staging, 0, size * 4);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(staging.getMappedRange()).slice();
      staging.unmap();
      staging.destroy();
      return result;
    },

    // --- Forward ---

    matmul(a, b, M, K, N) {
      const pl = getOrCreatePipeline('matmul', matmulWGSL);
      const u = uniformBuf([M, K, N, 0]);
      const out = emptyBuf(M * N);
      dispatch(pl, [u, a, b, out], Math.ceil(M / 16), Math.ceil(N / 16));
      u.destroy();
      return out;
    },

    matmulWT(x, w, M, K, N) {
      const pl = getOrCreatePipeline('matmulWT', matmulWTWGSL);
      const u = uniformBuf([M, K, N, 0]);
      const out = emptyBuf(M * N);
      dispatch(pl, [u, x, w, out], Math.ceil(M / 16), Math.ceil(N / 16));
      u.destroy();
      return out;
    },

    add(a, b, size) {
      const pl = getOrCreatePipeline('add', binary('out[i] = a[i] + b[i];'));
      const out = emptyBuf(size);
      dispatch(pl, [a, b, out], wg(size));
      return out;
    },

    relu(x, size) {
      const pl = getOrCreatePipeline('relu', unary('out[i] = max(0.0, a[i]);'));
      const out = emptyBuf(size);
      dispatch(pl, [x, out], wg(size));
      return out;
    },

    log(x, size) {
      const pl = getOrCreatePipeline('log', unary('out[i] = log(a[i]);'));
      const out = emptyBuf(size);
      dispatch(pl, [x, out], wg(size));
      return out;
    },

    exp(x, size) {
      const pl = getOrCreatePipeline('exp', unary('out[i] = exp(a[i]);'));
      const out = emptyBuf(size);
      dispatch(pl, [x, out], wg(size));
      return out;
    },

    scale(x, s, size) {
      const code = `
struct P { s: f32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= arrayLength(&out)) { return; } out[i] = a[i] * p.s;
}`;
      const pl = getOrCreatePipeline('scale', code);
      const u = uniformBuf([f2u(s), 0]);
      const out = emptyBuf(size);
      dispatch(pl, [u, x, out], wg(size));
      u.destroy();
      return out;
    },

    negate(x, size) {
      const pl = getOrCreatePipeline('negate', unary('out[i] = -a[i];'));
      const out = emptyBuf(size);
      dispatch(pl, [x, out], wg(size));
      return out;
    },

    embedding(table, ids, vocabSize, embdDim, seqLen) {
      const pl = getOrCreatePipeline('embedding', embeddingWGSL);
      const u = uniformBuf([embdDim, seqLen]);
      const idsBuf = uploadIds(ids);
      const out = emptyBuf(seqLen * embdDim);
      dispatch(pl, [u, table, idsBuf, out], wg(seqLen * embdDim));
      u.destroy(); idsBuf.destroy();
      return out;
    },

    rmsnorm(x, rows, cols) {
      const pl = getOrCreatePipeline('rmsnorm', rmsnormWGSL);
      const u = uniformBuf([rows, cols]);
      const out = emptyBuf(rows * cols);
      dispatch(pl, [u, x, out], wg(rows));
      u.destroy();
      return out;
    },

    softmax(x, rows, cols) {
      const pl = getOrCreatePipeline('softmax', softmaxWGSL);
      const u = uniformBuf([rows, cols]);
      const out = emptyBuf(rows * cols);
      dispatch(pl, [u, x, out], wg(rows));
      u.destroy();
      return out;
    },

    crossEntropyLoss(logits, targets, rows, cols) {
      const pl = getOrCreatePipeline('crossEntropy', crossEntropyWGSL);
      const u = uniformBuf([rows, cols]);
      const targetsBuf = uploadIds(targets);
      const probs = emptyBuf(rows * cols);
      const loss = emptyBuf(1);
      dispatch(pl, [u, logits, targetsBuf, probs, loss], 1);
      u.destroy(); targetsBuf.destroy();
      return { lossHandle: loss, softmaxOut: probs };
    },

    // --- Data movement ---

    sliceCols(x, rows, totalCols, colStart, width) {
      const pl = getOrCreatePipeline('sliceCols', sliceColsWGSL);
      const u = uniformBuf([rows, totalCols, colStart, width]);
      const out = emptyBuf(rows * width);
      dispatch(pl, [u, x, out], wg(rows * width));
      u.destroy();
      return out;
    },

    concatCols(handles, rows, widths) {
      const totalCols = widths.reduce((a, b) => a + b, 0);
      const out = emptyBuf(rows * totalCols);
      const pl = getOrCreatePipeline('copyCols', copyColsWGSL);
      let colOff = 0;
      for (let i = 0; i < handles.length; i++) {
        const u = uniformBuf([rows, widths[i], totalCols, colOff]);
        dispatch(pl, [u, handles[i], out], wg(rows * widths[i]));
        u.destroy();
        colOff += widths[i];
      }
      return out;
    },

    stackRows(handles, n, cols) {
      const out = emptyBuf(n * cols);
      const enc = device.createCommandEncoder();
      for (let i = 0; i < n; i++) {
        enc.copyBufferToBuffer(handles[i], 0, out, i * cols * 4, cols * 4);
      }
      device.queue.submit([enc.finish()]);
      return out;
    },

    transpose2D(x, rows, cols) {
      const pl = getOrCreatePipeline('transpose2D', transpose2DWGSL);
      const u = uniformBuf([rows, cols]);
      const out = emptyBuf(cols * rows);
      dispatch(pl, [u, x, out], wg(rows * cols));
      u.destroy();
      return out;
    },

    // --- Backward ---

    matmulBackward(dC, a, b, M, K, N) {
      // dA = dC[M,N] @ B^T => use matmul with B transposed
      // dA[i,k] = sum_j dC[i,j] * B[k,j] => matmulWT(dC, B^T)... actually:
      // dA = matmulWT where "W" = B reshaped. B is [K,N], dC is [M,N].
      // dA[i,k] = sum_j dC[i,j] * B[k,j]. B is [K,N] => B "row" k has N elems.
      // This is exactly matmulWT(dC[M,N], B[K,N]) = dA[M,K]
      const dA = this.matmulWT(dC, b, M, N, K);
      // dB = A^T @ dC => transpose A then regular matmul
      const aT = this.transpose2D(a, M, K);
      const dB = this.matmul(aT, dC, K, M, N);
      return { dA, dB };
    },

    matmulWTBackward(dOut, x, w, M, K, N) {
      // dx = dOut[M,N] @ W[N,K] (regular matmul)
      const dx = this.matmul(dOut, w, M, N, K);
      // dw = dOut^T[N,M] @ X[M,K] (regular matmul)
      const dOutT = this.transpose2D(dOut, M, N);
      const dw = this.matmul(dOutT, x, N, M, K);
      return { dx, dw };
    },

    addBackward(dC, size) {
      // Need separate copies for accumulation
      const pl = getOrCreatePipeline('copy', unary('out[i] = a[i];'));
      const dA = emptyBuf(size);
      const dB = emptyBuf(size);
      dispatch(pl, [dC, dA], wg(size));
      dispatch(pl, [dC, dB], wg(size));
      return { dA, dB };
    },

    reluBackward(dOut, x, size) {
      const pl = getOrCreatePipeline('reluBwd', reluBackwardWGSL);
      const dx = emptyBuf(size);
      dispatch(pl, [dOut, x, dx], wg(size));
      return dx;
    },

    logBackward(dOut, x, size) {
      const pl = getOrCreatePipeline('logBwd', binary('out[i] = a[i] / b[i];'));
      const dx = emptyBuf(size);
      dispatch(pl, [dOut, x, dx], wg(size));
      return dx;
    },

    expBackward(dOut, expOut, size) {
      const pl = getOrCreatePipeline('expBwd', binary('out[i] = a[i] * b[i];'));
      const dx = emptyBuf(size);
      dispatch(pl, [dOut, expOut, dx], wg(size));
      return dx;
    },

    scaleBackward(dOut, s, size) {
      return this.scale(dOut, s, size);
    },

    negateBackward(dOut, size) {
      return this.negate(dOut, size);
    },

    embeddingBackward(dOut, ids, vocabSize, embdDim, seqLen) {
      const pl = getOrCreatePipeline('embeddingBwd', embeddingBackwardWGSL);
      const u = uniformBuf([vocabSize, embdDim, seqLen, 0]);
      const idsBuf = uploadIds(ids);
      const dtable = emptyBuf(vocabSize * embdDim);
      dispatch(pl, [u, dOut, idsBuf, dtable], wg(seqLen * embdDim));
      u.destroy(); idsBuf.destroy();
      return dtable;
    },

    rmsnormBackward(dOut, x, rows, cols) {
      const pl = getOrCreatePipeline('rmsnormBwd', rmsnormBackwardWGSL);
      const u = uniformBuf([rows, cols]);
      const dx = emptyBuf(rows * cols);
      dispatch(pl, [u, dOut, x, dx], wg(rows));
      u.destroy();
      return dx;
    },

    softmaxBackward(dOut, softmaxOut, rows, cols) {
      const pl = getOrCreatePipeline('softmaxBwd', softmaxBackwardWGSL);
      const u = uniformBuf([rows, cols]);
      const dx = emptyBuf(rows * cols);
      dispatch(pl, [u, dOut, softmaxOut, dx], wg(rows));
      u.destroy();
      return dx;
    },

    crossEntropyBackward(softmaxOut, targets, rows, cols) {
      const pl = getOrCreatePipeline('crossEntropyBwd', crossEntropyBackwardWGSL);
      const u = uniformBuf([rows, cols]);
      const targetsBuf = uploadIds(targets);
      const dlogits = emptyBuf(rows * cols);
      dispatch(pl, [u, softmaxOut, targetsBuf, dlogits], wg(rows * cols));
      u.destroy(); targetsBuf.destroy();
      return dlogits;
    },

    sliceColsBackward(dOut, rows, totalCols, colStart, width) {
      const pl = getOrCreatePipeline('sliceColsBwd', sliceColsBackwardWGSL);
      const u = uniformBuf([rows, totalCols, colStart, width]);
      const dx = emptyBuf(rows * totalCols);
      dispatch(pl, [u, dOut, dx], wg(rows * width));
      u.destroy();
      return dx;
    },

    splitCols(dOut, rows, widths) {
      const totalCols = widths.reduce((a, b) => a + b, 0);
      const pl = getOrCreatePipeline('extractCols', extractColsWGSL);
      const results = [];
      let colOff = 0;
      for (const w of widths) {
        const u = uniformBuf([rows, totalCols, w, colOff]);
        const out = emptyBuf(rows * w);
        dispatch(pl, [u, dOut, out], wg(rows * w));
        u.destroy();
        results.push(out);
        colOff += w;
      }
      return results;
    },

    splitRows(handle, n, cols) {
      const results = [];
      const enc = device.createCommandEncoder();
      for (let i = 0; i < n; i++) {
        const buf = emptyBuf(cols);
        enc.copyBufferToBuffer(handle, i * cols * 4, buf, 0, cols * 4);
        results.push(buf);
      }
      device.queue.submit([enc.finish()]);
      return results;
    },

    // --- Optimizer ---

    adamUpdate(params, grads, mBuf, vBuf, lr, beta1, beta2, eps, step, count) {
      const pl = getOrCreatePipeline('adam', adamWGSL);
      const bc1 = 1 - Math.pow(beta1, step);
      const bc2 = 1 - Math.pow(beta2, step);
      const u = uniformBuf([f2u(lr), f2u(beta1), f2u(beta2), f2u(eps), f2u(bc1), f2u(bc2), count, 0]);
      dispatch(pl, [u, params, grads, mBuf, vBuf], wg(count));
      u.destroy();
    },

    // --- Utilities ---

    zeros(size) { return emptyBuf(size); },

    randn(size, std) {
      const data = new Float32Array(size);
      for (let i = 0; i < size; i += 2) {
        const u1 = Math.random(), u2 = Math.random();
        const r = Math.sqrt(-2 * Math.log(u1));
        data[i] = r * Math.cos(2 * Math.PI * u2) * std;
        if (i + 1 < size) data[i + 1] = r * Math.sin(2 * Math.PI * u2) * std;
      }
      return createGPUBuffer(data);
    },

    accumulate(dst, src, size) {
      const pl = getOrCreatePipeline('accumulate', accumulateWGSL);
      dispatch(pl, [dst, src], wg(size));
    },

    fill(handle, value, size) {
      const pl = getOrCreatePipeline('fill', fillWGSL);
      const u = uniformBuf([f2u(value), size]);
      dispatch(pl, [u, handle], wg(size));
      u.destroy();
    },
  };
}
