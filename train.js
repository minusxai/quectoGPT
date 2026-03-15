// quectoGPT training — GPT model definition + training loop
// Works with both CPU and WebGPU backends
// Node: node train.js [--backend=cpu|webgpu] [--steps=100]
// Browser: imported by index.html

import { Tensor, backward, zeroGrad, clearTape, ops, Adam } from './quectrograd/index.js';
import { initCPU } from './quectrograd/backend_cpu.js';

// --- Hyperparameters (matching microgpt.py) ---
const N_LAYER = 1;
const N_EMBD = 16;
const BLOCK_SIZE = 16;
const N_HEAD = 4;
const HEAD_DIM = N_EMBD / N_HEAD;

// --- Seeded RNG (for reproducibility) ---
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function seededGauss(rng, std = 0.08) {
  const u1 = rng(), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * std;
}

// --- Tokenizer ---
export function buildTokenizer(docs) {
  const uchars = [...new Set(docs.join(''))].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;
  const encode = (str) => [BOS, ...str.split('').map(c => uchars.indexOf(c)), BOS];
  const decode = (ids) => ids.filter(id => id !== BOS).map(id => uchars[id]).join('');
  return { uchars, BOS, vocabSize, encode, decode };
}

// --- Parameter initialization ---
function initParams(vocabSize, backend, rng) {
  const matrix = (rows, cols, std = 0.08) => {
    const data = new Float32Array(rows * cols);
    for (let i = 0; i < data.length; i++) data[i] = seededGauss(rng, std);
    return Tensor.from(data, [rows, cols], backend, { requiresGrad: true });
  };

  const stateDict = {
    wte: matrix(vocabSize, N_EMBD),
    wpe: matrix(BLOCK_SIZE, N_EMBD),
    lm_head: matrix(vocabSize, N_EMBD),
  };

  for (let i = 0; i < N_LAYER; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.attn_wk`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.attn_wv`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.attn_wo`] = matrix(N_EMBD, N_EMBD);
    stateDict[`layer${i}.mlp_fc1`] = matrix(4 * N_EMBD, N_EMBD);
    stateDict[`layer${i}.mlp_fc2`] = matrix(N_EMBD, 4 * N_EMBD);
  }

  return stateDict;
}

function getAllParams(stateDict) {
  return Object.values(stateDict);
}

// --- Single-token GPT forward (used for both training and inference) ---
// Processes one token at position `pos`, using KV cache for attention.
// Returns the output embedding [1, N_EMBD] after all layers.
function gptForwardToken(tokenId, posId, kvCache, stateDict, backend) {
  const tokenIds = new Int32Array([tokenId]);
  const posIds = new Int32Array([posId]);

  let row = ops.embedding(stateDict.wte, tokenIds);      // [1, N_EMBD]
  const posEmb = ops.embedding(stateDict.wpe, posIds);    // [1, N_EMBD]
  row = ops.add(row, posEmb);
  row = ops.rmsnorm(row);

  for (let li = 0; li < N_LAYER; li++) {
    const rowResidual = row;
    row = ops.rmsnorm(row);

    // QKV projections using matmulWT (matches Python's linear)
    const q = ops.matmulWT(row, stateDict[`layer${li}.attn_wq`]); // [1, N_EMBD]
    const k = ops.matmulWT(row, stateDict[`layer${li}.attn_wk`]);
    const v = ops.matmulWT(row, stateDict[`layer${li}.attn_wv`]);

    kvCache[li].keys.push(k);
    kvCache[li].values.push(v);

    // Multi-head attention
    const headOuts = [];
    for (let h = 0; h < N_HEAD; h++) {
      const hs = h * HEAD_DIM;

      // Extract head slices (differentiable)
      const qH = ops.sliceCols(q, hs, HEAD_DIM); // [1, HEAD_DIM]
      const numKeys = kvCache[li].keys.length;

      // Compute attention scores against all cached keys
      const scoreTerms = [];
      for (let t = 0; t < numKeys; t++) {
        const kT = ops.sliceCols(kvCache[li].keys[t], hs, HEAD_DIM); // [1, HEAD_DIM]
        const kTrans = ops.transpose2D(kT); // [HEAD_DIM, 1]
        const score = ops.matmul(qH, kTrans); // [1, 1]
        scoreTerms.push(ops.scale(score, 1.0 / Math.sqrt(HEAD_DIM)));
      }

      // Concatenate scores into [1, numKeys] and softmax
      const scoresRow = ops.concatCols(scoreTerms);
      const attnWeights = ops.softmax(scoresRow); // [1, numKeys]

      // Weighted sum of values: attn_weights[1, numKeys] @ V_stacked[numKeys, HEAD_DIM]
      const vSlices = kvCache[li].values.map(vv => ops.sliceCols(vv, hs, HEAD_DIM));
      const vStack = ops.stackRows(vSlices); // [numKeys, HEAD_DIM]
      const headOut = ops.matmul(attnWeights, vStack); // [1, HEAD_DIM]
      headOuts.push(headOut);
    }

    // Concat heads → [1, N_EMBD]
    const xAttn = ops.concatCols(headOuts);

    // Output projection + residual
    row = ops.matmulWT(xAttn, stateDict[`layer${li}.attn_wo`]);
    row = ops.add(row, rowResidual);

    // MLP block
    const mlpResidual = row;
    row = ops.rmsnorm(row);
    row = ops.matmulWT(row, stateDict[`layer${li}.mlp_fc1`]); // [1, 4*N_EMBD]
    row = ops.relu(row);
    row = ops.matmulWT(row, stateDict[`layer${li}.mlp_fc2`]); // [1, N_EMBD]
    row = ops.add(row, mlpResidual);
  }

  return row; // [1, N_EMBD]
}

// --- Full forward pass for training: process a document ---
function gptForward(tokens, stateDict, backend) {
  const seqLen = tokens.length - 1;
  const n = Math.min(BLOCK_SIZE, seqLen);

  const kvCache = [];
  for (let li = 0; li < N_LAYER; li++) kvCache.push({ keys: [], values: [] });

  const outputRows = [];
  const targetIds = new Int32Array(n);

  for (let pos = 0; pos < n; pos++) {
    targetIds[pos] = tokens[pos + 1];
    const row = gptForwardToken(tokens[pos], pos, kvCache, stateDict, backend);
    outputRows.push(row);
  }

  // Stack all output rows → [n, N_EMBD]
  const output = ops.stackRows(outputRows);

  // Logits: [n, N_EMBD] @ lm_head^T → [n, vocabSize]
  const logits = ops.matmulWT(output, stateDict.lm_head);

  // Cross-entropy loss
  const loss = ops.crossEntropyLoss(logits, targetIds);

  return { loss, logits, n };
}

// --- Inference ---
export async function inference(stateDict, tokenizer, backend, opts = {}) {
  const temperature = opts.temperature ?? 0.5;
  const numSamples = opts.numSamples ?? 20;
  const rng = opts.rng ?? Math.random;
  const samples = [];

  for (let s = 0; s < numSamples; s++) {
    let tokenId = tokenizer.BOS;
    const generated = [];

    const kvCache = [];
    for (let li = 0; li < N_LAYER; li++) kvCache.push({ keys: [], values: [] });

    for (let pos = 0; pos < BLOCK_SIZE; pos++) {
      clearTape();

      const row = gptForwardToken(tokenId, pos, kvCache, stateDict, backend);
      const logits = ops.matmulWT(row, stateDict.lm_head); // [1, vocabSize]

      // Apply temperature and sample
      const logitsArr = await logits.toArray();
      if (temperature !== 1.0) {
        for (let i = 0; i < logitsArr.length; i++) logitsArr[i] /= temperature;
      }
      // Softmax
      let maxVal = -Infinity;
      for (let i = 0; i < logitsArr.length; i++) if (logitsArr[i] > maxVal) maxVal = logitsArr[i];
      const probs = new Float32Array(logitsArr.length);
      let sum = 0;
      for (let i = 0; i < logitsArr.length; i++) {
        probs[i] = Math.exp(logitsArr[i] - maxVal);
        sum += probs[i];
      }
      for (let i = 0; i < probs.length; i++) probs[i] /= sum;

      // Weighted random choice
      const r = rng();
      let cumul = 0;
      let chosen = probs.length - 1;
      for (let i = 0; i < probs.length; i++) {
        cumul += probs[i];
        if (r < cumul) { chosen = i; break; }
      }

      if (chosen === tokenizer.BOS) break;
      generated.push(tokenizer.uchars[chosen]);
      tokenId = chosen;
    }

    samples.push(generated.join(''));
  }

  return samples;
}

// --- Training generator (used by both Node CLI and browser) ---
export async function* train(backend, docs, opts = {}) {
  const numSteps = opts.steps ?? 100;
  const learningRate = opts.lr ?? 0.01;
  const seed = opts.seed ?? 42;
  const rng = mulberry32(seed);

  // Shuffle docs with seeded RNG
  const shuffledDocs = [...docs];
  for (let i = shuffledDocs.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffledDocs[i], shuffledDocs[j]] = [shuffledDocs[j], shuffledDocs[i]];
  }

  const tokenizer = buildTokenizer(shuffledDocs);
  const stateDict = initParams(tokenizer.vocabSize, backend, rng);
  const params = getAllParams(stateDict);
  const paramCount = params.reduce((s, p) => s + p.numel(), 0);

  const optimizer = new Adam(params, {
    lr: learningRate,
    beta1: 0.85,
    beta2: 0.99,
    eps: 1e-8,
  });

  yield { type: 'init', vocabSize: tokenizer.vocabSize, paramCount, backend: backend.name };

  for (let step = 0; step < numSteps; step++) {
    clearTape();
    zeroGrad(params);

    const doc = shuffledDocs[step % shuffledDocs.length];
    const tokens = tokenizer.encode(doc);
    const { loss, n } = gptForward(tokens, stateDict, backend);

    backward(loss);

    const lrT = learningRate * (1 - step / numSteps);
    optimizer.step(lrT);

    const lossVal = (await loss.toArray())[0];
    yield { type: 'step', step: step + 1, loss: lossVal, n, totalSteps: numSteps };
  }

  // Inference
  clearTape();
  const samples = await inference(stateDict, tokenizer, backend, {
    temperature: 0.5,
    numSamples: 20,
    rng,
  });

  yield { type: 'done', samples, tokenizer, stateDict };
}

// --- CLI entry point (Node + Deno) ---
async function main() {
  const isDeno = typeof Deno !== 'undefined';
  const args = isDeno ? Deno.args : process.argv.slice(2);
  const getArg = (name, def) => {
    const a = args.find(x => x.startsWith(`--${name}=`));
    return a ? a.split('=')[1] : def;
  };

  const backendName = getArg('backend', 'cpu');
  const steps = parseInt(getArg('steps', '100'));

  let backend;
  if (backendName === 'webgpu') {
    const { initWebGPU } = await import('./quectrograd/backend_webgpu.js');
    backend = await initWebGPU();
  } else {
    backend = initCPU();
  }

  // Load data
  const text = isDeno
    ? Deno.readTextFileSync('input.txt')
    : (await import('fs')).readFileSync('input.txt', 'utf-8');
  const docs = text.split('\n').filter(l => l.trim());

  console.log(`num docs: ${docs.length}`);
  console.log(`backend: ${backend.name}`);

  const gen = train(backend, docs, { steps });

  const write = isDeno
    ? (s) => Deno.stdout.writeSync(new TextEncoder().encode(s))
    : (s) => process.stdout.write(s);

  for await (const event of gen) {
    if (event.type === 'init') {
      console.log(`vocab size: ${event.vocabSize}`);
      console.log(`num params: ${event.paramCount}`);
    } else if (event.type === 'step') {
      write(`\rstep ${String(event.step).padStart(4)} / ${String(event.totalSteps).padStart(4)} | loss ${event.loss.toFixed(4)}`);
    } else if (event.type === 'done') {
      console.log('\n--- inference (new, hallucinated names) ---');
      event.samples.forEach((s, i) => console.log(`sample ${String(i + 1).padStart(2)}: ${s}`));
    }
  }
}

// Run if executed directly
const isDeno = typeof Deno !== 'undefined';
const isNode = typeof process !== 'undefined' && process.versions?.node;
if (isDeno) {
  main().catch(console.error);
} else if (isNode) {
  const url = await import('url');
  if (import.meta.url === url.pathToFileURL(process.argv[1]).href) {
    main().catch(console.error);
  }
}
