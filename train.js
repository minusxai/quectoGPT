// quectoGPT training — training loop with jax-js
// Deno: deno run --allow-read --allow-net --allow-env train.js --dataset=names [--backend=cpu] [--steps=200] [--model=medium] [--batch=8]
// Browser: imported by index.html

import { init, defaultDevice, numpy as np, nn, valueAndGrad, random, tree, blockUntilReady } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';
import { MODEL_CONFIGS, resolveConfig, initParams, loss, inference } from './gpt.js';
import { buildBPETokenizer } from './bpe.js';

export { MODEL_CONFIGS, buildBPETokenizer };

// --- Seeded RNG (for batch sampling — JS side) ---
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// --- LR schedule: warmup -> constant -> linear decay ---
function lrMultiplier(step, totalSteps) {
  const warmup = Math.min(10, Math.floor(totalSteps * 0.1));
  const decayStart = Math.floor(totalSteps * 0.6);
  if (step < warmup) return (step + 1) / warmup;
  if (step < decayStart) return 1.0;
  return (totalSteps - step) / (totalSteps - decayStart);
}

// --- Weight serialization helpers ---
function flattenParams(params) {
  const leaves = tree.leaves(params);
  const total = leaves.reduce((s, l) => s + l.size, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const leaf of leaves) {
    out.set(leaf.dataSync(), offset);
    offset += leaf.size;
  }
  return out;
}

function loadWeights(flat, templateParams) {
  let offset = 0;
  return tree.map(templateParams, leaf => {
    const chunk = flat.slice(offset, offset + leaf.size);
    offset += leaf.size;
    const newLeaf = np.array(chunk).reshape(leaf.shape);
    leaf.dispose();
    return newLeaf;
  });
}

// --- Sample a batch of random windows from token data ---
function sampleBatch(tokenData, batchSize, blockSize, rng) {
  const maxStart = tokenData.length - blockSize - 1;
  const inputBuf = new Int32Array(batchSize * blockSize);
  const targetBuf = new Int32Array(batchSize * blockSize);

  for (let b = 0; b < batchSize; b++) {
    const start = Math.floor(rng() * maxStart);
    for (let i = 0; i < blockSize; i++) {
      inputBuf[b * blockSize + i] = tokenData[start + i];
      targetBuf[b * blockSize + i] = tokenData[start + i + 1];
    }
  }

  return { inputBuf, targetBuf };
}

// --- Training generator ---
// trainData/valData: Uint16Array of pre-tokenized tokens
// tokenizer: { encode, decode, vocabSize, BOS, EOS } from buildBPETokenizer
export async function* train(backendName, trainData, valData, tokenizer, opts = {}) {
  const modelName = opts.model ?? 'nano';
  const cfg = resolveConfig(modelName);
  if (opts.blockSize) cfg.blockSize = opts.blockSize; // override context window
  const numSteps = opts.steps ?? cfg.steps;
  const batchSize = opts.batchSize ?? 8;
  const baseLR = opts.lr ?? cfg.lr;
  const seed = opts.seed ?? 42;
  const valEvery = opts.valEvery ?? Math.max(10, Math.floor(numSteps / 20)); // ~20 val evals, min every 10 steps
  const valBatches = opts.valBatches ?? 4; // average val loss over this many batches
  const jsRng = mulberry32(seed);
  const valRng = mulberry32(seed + 999);

  // Init jax-js params
  const rngKey = random.key(seed);
  let params = await initParams(tokenizer.vocabSize, cfg, rngKey);

  // Count params
  const paramLeaves = tree.leaves(params);
  const paramCount = paramLeaves.reduce((s, p) => s + p.size, 0);

  // Load server weights if provided
  if (opts.initialWeights?.length === paramCount) {
    params = loadWeights(new Float32Array(opts.initialWeights), params);
  }

  // Expose live flat weights via handle so caller can snapshot them
  const h = opts.handle;
  if (h) h.getFlat = () => flattenParams(params);

  // Optimizer — use schedule for LR
  const lrSchedule = (count) => baseLR * lrMultiplier(count, numSteps);
  const solver = adam(lrSchedule, { b1: 0.85, b2: 0.99 });
  let optState = solver.init(tree.ref(params));

  yield {
    type: 'init', vocabSize: tokenizer.vocabSize, paramCount,
    backend: backendName, model: modelName, cfg, batchSize,
    trainTokens: trainData.length, valTokens: valData ? valData.length : 0,
  };

  const seqLen = cfg.blockSize;

  for (let step = 0; step < numSteps; step++) {
    // Sample batch from training data
    const { inputBuf, targetBuf } = sampleBatch(trainData, batchSize, seqLen, jsRng);

    const inputIds = np.array(inputBuf, { dtype: np.int32 }).reshape([batchSize, seqLen]);
    const posIds = np.tile(np.arange(seqLen).astype(np.int32), [batchSize, 1]);
    const targetIds = np.array(targetBuf, { dtype: np.int32 }).reshape([batchSize, seqLen]);

    // Pre-compute oneHot matrices OUTSIDE valueAndGrad (avoids tracing issues)
    const tokenOH = nn.oneHot(inputIds, tokenizer.vocabSize);   // [B, seqLen, vocabSize]
    const posOH = nn.oneHot(posIds, cfg.blockSize);              // [B, seqLen, blockSize]
    const targetOH = nn.oneHot(targetIds, tokenizer.vocabSize);  // [B, seqLen, vocabSize]

    // Forward + backward
    const lossFn = (p) => loss(p, cfg, tokenOH.ref, posOH.ref, targetOH.ref, seqLen, null, true);
    const [lossVal, grads] = valueAndGrad(lossFn)(tree.ref(params));
    const lossScalar = lossVal.item();

    // Optimizer step
    const oldParams = params;
    const oldOptState = optState;
    const [updates, newOptState] = solver.update(grads, oldOptState, tree.ref(oldParams));
    params = applyUpdates(oldParams, updates);
    optState = newOptState;
    await blockUntilReady(params);

    // Dispose per-step intermediates
    tokenOH.dispose();
    posOH.dispose();
    targetOH.dispose();

    const event = { type: 'step', step: step + 1, loss: lossScalar, totalSteps: numSteps, batchSize };

    // Val loss evaluation
    if (valData && valData.length > seqLen + 1 && (step + 1) % valEvery === 0) {
      let valLossSum = 0;
      for (let vb = 0; vb < valBatches; vb++) {
        const val = sampleBatch(valData, batchSize, seqLen, valRng);
        const vInput = np.array(val.inputBuf, { dtype: np.int32 }).reshape([batchSize, seqLen]);
        const vPos = np.tile(np.arange(seqLen).astype(np.int32), [batchSize, 1]);
        const vTarget = np.array(val.targetBuf, { dtype: np.int32 }).reshape([batchSize, seqLen]);
        const vTokOH = nn.oneHot(vInput, tokenizer.vocabSize);
        const vPosOH = nn.oneHot(vPos, cfg.blockSize);
        const vTargOH = nn.oneHot(vTarget, tokenizer.vocabSize);
        const vLoss = loss(tree.ref(params), cfg, vTokOH, vPosOH, vTargOH, seqLen, null, true);
        valLossSum += vLoss.item();
        // oneHot tensors are consumed by loss() — no manual dispose needed
      }
      event.valLoss = valLossSum / valBatches;

      // Generate samples at each val eval
      const sampleKey = random.key(seed + step + 100);
      const sampleOpts = { temperature: opts.temperature ?? 0.8, numSamples: 10 };
      if (opts.prompt) sampleOpts.prompt = opts.prompt;
      event.samples = await inference(params, cfg, tokenizer, sampleKey, sampleOpts);
    }

    yield event;
  }

  // Inference
  const inferKey = random.key(seed + 1);
  const inferOpts = {
    temperature: opts.temperature ?? 0.5,
    numSamples: opts.numSamples ?? 20,
  };
  if (opts.prompt) inferOpts.prompt = opts.prompt;
  const samples = await inference(params, cfg, tokenizer, inferKey, inferOpts);

  yield { type: 'done', samples, tokenizer };
}

// --- CLI entry point ---
async function main() {
  const isDeno = typeof Deno !== 'undefined';
  const args = isDeno ? Deno.args : process.argv.slice(2);
  const getArg = (name, def) => {
    const a = args.find(x => x.startsWith(`--${name}=`));
    return a ? a.split('=')[1] : def;
  };

  const dataset = getArg('dataset', 'names');
  const modelName = getArg('model', 'nano');
  const backendArg = getArg('backend', 'cpu');
  const steps = getArg('steps', null);
  const batch = getArg('batch', '8');
  const blocksize = getArg('blocksize', null);
  const valevery = getArg('valevery', null);
  const prompt = getArg('prompt', null);

  // Initialize jax-js
  const devices = await init();
  const device = backendArg === 'webgpu' ? 'webgpu' : 'wasm';
  if (!devices.includes(device)) {
    console.error(`Backend "${device}" not available. Available: ${devices.join(', ')}`);
    if (typeof Deno !== 'undefined') Deno.exit(1); else process.exit(1);
  }
  defaultDevice(device);

  // Load tokenizer
  const tokText = isDeno
    ? Deno.readTextFileSync(`data/${dataset}.tok.json`)
    : (await import('fs')).readFileSync(`data/${dataset}.tok.json`, 'utf-8');
  const tokJson = JSON.parse(tokText);
  const tokenizer = buildBPETokenizer(tokJson.merges);

  // Load pre-tokenized binary data
  let trainData, valData;
  if (isDeno) {
    trainData = new Uint16Array(Deno.readFileSync(`data/${dataset}.train.bin`).buffer);
    valData = new Uint16Array(Deno.readFileSync(`data/${dataset}.val.bin`).buffer);
  } else {
    const fs = await import('fs');
    const trainBuf = fs.readFileSync(`data/${dataset}.train.bin`);
    const valBuf = fs.readFileSync(`data/${dataset}.val.bin`);
    trainData = new Uint16Array(trainBuf.buffer, trainBuf.byteOffset, trainBuf.byteLength / 2);
    valData = new Uint16Array(valBuf.buffer, valBuf.byteOffset, valBuf.byteLength / 2);
  }

  const backendDisplay = device === 'wasm' ? 'cpu (wasm)' : device;
  console.log(`dataset: ${dataset}`);
  console.log(`model: ${modelName}`);
  console.log(`backend: ${backendDisplay}`);
  console.log(`train tokens: ${trainData.length.toLocaleString()}`);
  console.log(`val tokens: ${valData.length.toLocaleString()}`);

  const trainOpts = { model: modelName, batchSize: parseInt(batch) };
  if (steps) trainOpts.steps = parseInt(steps);
  if (blocksize) trainOpts.blockSize = parseInt(blocksize);
  if (valevery) trainOpts.valEvery = parseInt(valevery);
  if (prompt) trainOpts.prompt = prompt;

  const gen = train(device, trainData, valData, tokenizer, trainOpts);

  const write = isDeno
    ? (s) => Deno.stdout.writeSync(new TextEncoder().encode(s))
    : (s) => process.stdout.write(s);

  for await (const event of gen) {
    if (event.type === 'init') {
      console.log(`vocab size: ${event.vocabSize}`);
      console.log(`batch size: ${event.batchSize}`);
      console.log(`num params: ${event.paramCount.toLocaleString()}`);
    } else if (event.type === 'step') {
      let line = `\rstep ${String(event.step).padStart(4)} / ${String(event.totalSteps).padStart(4)} | loss ${event.loss.toFixed(4)}`;
      if (event.valLoss !== undefined) {
        line += ` | val ${event.valLoss.toFixed(4)}`;
      }
      write(line);
    } else if (event.type === 'done') {
      console.log('\n--- generated samples ---');
      event.samples.forEach((s, i) => console.log(`sample ${String(i + 1).padStart(2)}: ${s}`));
    }
  }
}

// Run if executed directly
if (typeof Deno !== 'undefined') {
  if (import.meta.main) main().catch(console.error);
} else if (typeof process !== 'undefined' && process.versions?.node) {
  const url = await import('url');
  if (import.meta.url === url.pathToFileURL(process.argv[1]).href) {
    main().catch(console.error);
  }
}
