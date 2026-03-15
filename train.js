// quectoGPT training — GPT model definition + training loop
// Works with both CPU and WebGPU backends
// Node: node train.js [--backend=cpu|webgpu] [--steps=200] [--model=medium]
// Deno: deno run --allow-read --unstable-webgpu train.js --backend=webgpu --model=medium
// Browser: imported by index.html

import { Tensor, backward, zeroGrad, clearTape, ops, Adam } from './quectrograd/index.js';
import { initCPU } from './quectrograd/backend_cpu.js';

// --- Model configs ---
export const MODEL_CONFIGS = {
  nano:   { nLayer: 1, nEmbd: 16,  blockSize: 16,  nHead: 4, lr: 0.01,  initStd: 0.08, steps: 100 },
  tiny:   { nLayer: 2, nEmbd: 64,  blockSize: 32,  nHead: 4, lr: 3e-3,  initStd: 0.04, steps: 200 },
  small:  { nLayer: 4, nEmbd: 128, blockSize: 64,  nHead: 4, lr: 1e-3,  initStd: 0.02, steps: 200 },
  medium: { nLayer: 6, nEmbd: 256, blockSize: 128, nHead: 8, lr: 3e-4,  initStd: 0.02, steps: 300 },
  large:  { nLayer: 8, nEmbd: 512, blockSize: 256, nHead: 8, lr: 1e-4,  initStd: 0.01, steps: 500 },
};

function resolveConfig(name) {
  const cfg = MODEL_CONFIGS[name];
  if (!cfg) throw new Error(`Unknown model config: ${name}. Choose from: ${Object.keys(MODEL_CONFIGS).join(', ')}`);
  return { ...cfg, headDim: cfg.nEmbd / cfg.nHead };
}

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
function initParams(vocabSize, cfg, backend, rng) {
  const matrix = (rows, cols, std = cfg.initStd) => {
    const data = new Float32Array(rows * cols);
    for (let i = 0; i < data.length; i++) data[i] = seededGauss(rng, std);
    return Tensor.from(data, [rows, cols], backend, { requiresGrad: true });
  };

  const stateDict = {
    wte: matrix(vocabSize, cfg.nEmbd),
    wpe: matrix(cfg.blockSize, cfg.nEmbd),
    lm_head: matrix(vocabSize, cfg.nEmbd),
  };

  for (let i = 0; i < cfg.nLayer; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(cfg.nEmbd, cfg.nEmbd);
    stateDict[`layer${i}.attn_wk`] = matrix(cfg.nEmbd, cfg.nEmbd);
    stateDict[`layer${i}.attn_wv`] = matrix(cfg.nEmbd, cfg.nEmbd);
    stateDict[`layer${i}.attn_wo`] = matrix(cfg.nEmbd, cfg.nEmbd);
    stateDict[`layer${i}.mlp_fc1`] = matrix(4 * cfg.nEmbd, cfg.nEmbd);
    stateDict[`layer${i}.mlp_fc2`] = matrix(cfg.nEmbd, 4 * cfg.nEmbd);
  }

  return stateDict;
}

function getAllParams(stateDict) {
  return Object.values(stateDict);
}

// --- Single-token GPT forward (used for both training and inference) ---
function gptForwardToken(tokenId, posId, kvCache, stateDict, cfg, backend) {
  const tokenIds = new Int32Array([tokenId]);
  const posIds = new Int32Array([posId]);

  let row = ops.embedding(stateDict.wte, tokenIds);
  const posEmb = ops.embedding(stateDict.wpe, posIds);
  row = ops.add(row, posEmb);
  row = ops.rmsnorm(row);

  for (let li = 0; li < cfg.nLayer; li++) {
    const rowResidual = row;
    row = ops.rmsnorm(row);

    const q = ops.matmulWT(row, stateDict[`layer${li}.attn_wq`]);
    const k = ops.matmulWT(row, stateDict[`layer${li}.attn_wk`]);
    const v = ops.matmulWT(row, stateDict[`layer${li}.attn_wv`]);

    kvCache[li].keys.push(k);
    kvCache[li].values.push(v);

    const headOuts = [];
    for (let h = 0; h < cfg.nHead; h++) {
      const hs = h * cfg.headDim;

      const qH = ops.sliceCols(q, hs, cfg.headDim);
      const numKeys = kvCache[li].keys.length;

      const scoreTerms = [];
      for (let t = 0; t < numKeys; t++) {
        const kT = ops.sliceCols(kvCache[li].keys[t], hs, cfg.headDim);
        const kTrans = ops.transpose2D(kT);
        const score = ops.matmul(qH, kTrans);
        scoreTerms.push(ops.scale(score, 1.0 / Math.sqrt(cfg.headDim)));
      }

      const scoresRow = ops.concatCols(scoreTerms);
      const attnWeights = ops.softmax(scoresRow);

      const vSlices = kvCache[li].values.map(vv => ops.sliceCols(vv, hs, cfg.headDim));
      const vStack = ops.stackRows(vSlices);
      const headOut = ops.matmul(attnWeights, vStack);
      headOuts.push(headOut);
    }

    const xAttn = ops.concatCols(headOuts);

    row = ops.matmulWT(xAttn, stateDict[`layer${li}.attn_wo`]);
    row = ops.add(row, rowResidual);

    const mlpResidual = row;
    row = ops.rmsnorm(row);
    row = ops.matmulWT(row, stateDict[`layer${li}.mlp_fc1`]);
    row = ops.relu(row);
    row = ops.matmulWT(row, stateDict[`layer${li}.mlp_fc2`]);
    row = ops.add(row, mlpResidual);
  }

  return row;
}

// --- Full forward pass for training: process a document ---
function gptForward(tokens, stateDict, cfg, backend) {
  const seqLen = tokens.length - 1;
  const n = Math.min(cfg.blockSize, seqLen);

  const kvCache = [];
  for (let li = 0; li < cfg.nLayer; li++) kvCache.push({ keys: [], values: [] });

  const outputRows = [];
  const targetIds = new Int32Array(n);

  for (let pos = 0; pos < n; pos++) {
    targetIds[pos] = tokens[pos + 1];
    const row = gptForwardToken(tokens[pos], pos, kvCache, stateDict, cfg, backend);
    outputRows.push(row);
  }

  const output = ops.stackRows(outputRows);
  const logits = ops.matmulWT(output, stateDict.lm_head);
  const loss = ops.crossEntropyLoss(logits, targetIds);

  return { loss, logits, n };
}

// --- Inference ---
export async function inference(stateDict, tokenizer, cfg, backend, opts = {}) {
  const temperature = opts.temperature ?? 0.5;
  const numSamples = opts.numSamples ?? 20;
  const rng = opts.rng ?? Math.random;
  const samples = [];

  for (let s = 0; s < numSamples; s++) {
    let tokenId = tokenizer.BOS;
    const generated = [];

    const kvCache = [];
    for (let li = 0; li < cfg.nLayer; li++) kvCache.push({ keys: [], values: [] });

    for (let pos = 0; pos < cfg.blockSize; pos++) {
      clearTape();

      const row = gptForwardToken(tokenId, pos, kvCache, stateDict, cfg, backend);
      const logits = ops.matmulWT(row, stateDict.lm_head);

      const logitsArr = await logits.toArray();
      if (temperature !== 1.0) {
        for (let i = 0; i < logitsArr.length; i++) logitsArr[i] /= temperature;
      }
      let maxVal = -Infinity;
      for (let i = 0; i < logitsArr.length; i++) if (logitsArr[i] > maxVal) maxVal = logitsArr[i];
      const probs = new Float32Array(logitsArr.length);
      let sum = 0;
      for (let i = 0; i < logitsArr.length; i++) {
        probs[i] = Math.exp(logitsArr[i] - maxVal);
        sum += probs[i];
      }
      for (let i = 0; i < probs.length; i++) probs[i] /= sum;

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
  const modelName = opts.model ?? 'nano';
  const cfg = resolveConfig(modelName);
  const numSteps = opts.steps ?? cfg.steps;
  const learningRate = opts.lr ?? cfg.lr;
  const seed = opts.seed ?? 42;
  const rng = mulberry32(seed);

  const shuffledDocs = [...docs];
  for (let i = shuffledDocs.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffledDocs[i], shuffledDocs[j]] = [shuffledDocs[j], shuffledDocs[i]];
  }

  const tokenizer = buildTokenizer(shuffledDocs);
  const stateDict = initParams(tokenizer.vocabSize, cfg, backend, rng);
  const params = getAllParams(stateDict);
  const paramCount = params.reduce((s, p) => s + p.numel(), 0);

  const optimizer = new Adam(params, {
    lr: learningRate,
    beta1: 0.85,
    beta2: 0.99,
    eps: 1e-8,
  });

  yield { type: 'init', vocabSize: tokenizer.vocabSize, paramCount, backend: backend.name, model: modelName, cfg };

  for (let step = 0; step < numSteps; step++) {
    clearTape();
    zeroGrad(params);

    const doc = shuffledDocs[step % shuffledDocs.length];
    const tokens = tokenizer.encode(doc);
    const { loss, n } = gptForward(tokens, stateDict, cfg, backend);

    backward(loss);

    const lrT = learningRate * (1 - step / numSteps);
    optimizer.step(lrT);

    const lossVal = (await loss.toArray())[0];
    yield { type: 'step', step: step + 1, loss: lossVal, n, totalSteps: numSteps };
  }

  clearTape();
  const samples = await inference(stateDict, tokenizer, cfg, backend, {
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

  const modelName = getArg('model', 'nano');
  const backendName = getArg('backend', 'cpu');
  const steps = getArg('steps', null);

  let backend;
  if (backendName === 'webgpu') {
    const { initWebGPU } = await import('./quectrograd/backend_webgpu.js');
    backend = await initWebGPU();
  } else {
    backend = initCPU();
  }

  const text = isDeno
    ? Deno.readTextFileSync('input.txt')
    : (await import('fs')).readFileSync('input.txt', 'utf-8');
  const docs = text.split('\n').filter(l => l.trim());

  console.log(`model: ${modelName}`);
  console.log(`num docs: ${docs.length}`);
  console.log(`backend: ${backend.name}`);

  const trainOpts = { model: modelName };
  if (steps) trainOpts.steps = parseInt(steps);

  const gen = train(backend, docs, trainOpts);

  const write = isDeno
    ? (s) => Deno.stdout.writeSync(new TextEncoder().encode(s))
    : (s) => process.stdout.write(s);

  for await (const event of gen) {
    if (event.type === 'init') {
      console.log(`vocab size: ${event.vocabSize}`);
      console.log(`num params: ${event.paramCount.toLocaleString()}`);
    } else if (event.type === 'step') {
      write(`\rstep ${String(event.step).padStart(4)} / ${String(event.totalSteps).padStart(4)} | loss ${event.loss.toFixed(4)}`);
    } else if (event.type === 'done') {
      console.log('\n--- inference (new, hallucinated names) ---');
      event.samples.forEach((s, i) => console.log(`sample ${String(i + 1).padStart(2)}: ${s}`));
    }
  }
}

// Run if executed directly (not when imported as a module)
if (typeof Deno !== 'undefined') {
  if (import.meta.main) main().catch(console.error);
} else if (typeof process !== 'undefined' && process.versions?.node) {
  const url = await import('url');
  if (import.meta.url === url.pathToFileURL(process.argv[1]).href) {
    main().catch(console.error);
  }
}
