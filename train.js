// quectoGPT training — training loop with jax-js
// Node: node train.js [--backend=cpu|webgpu] [--steps=200] [--model=medium]
// Deno: deno run --allow-read --allow-net --unstable-webgpu train.js --backend=webgpu --model=medium
// Browser: imported by index.html

import { init, defaultDevice, numpy as np, nn, valueAndGrad, random, tree, blockUntilReady } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';
import { MODEL_CONFIGS, resolveConfig, initParams, loss, inference } from './gpt.js';

export { MODEL_CONFIGS };

// --- Tokenizer (byte-level) ---
export function buildTokenizer() {
  const BOS = 256, EOS = 257;
  const vocabSize = 258;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const encode = (str) => [BOS, ...encoder.encode(str), EOS];
  const decode = (ids) => decoder.decode(new Uint8Array(ids.filter(id => id < 256)));
  return { BOS, EOS, vocabSize, encode, decode };
}

// --- Seeded RNG (for doc shuffling — JS side) ---
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

// --- Training generator ---
export async function* train(backendName, docs, opts = {}) {
  const modelName = opts.model ?? 'nano';
  const cfg = resolveConfig(modelName);
  const numSteps = opts.steps ?? cfg.steps;
  const batchSize = opts.batchSize ?? 8;
  const baseLR = opts.lr ?? cfg.lr;
  const seed = opts.seed ?? 42;
  const jsRng = mulberry32(seed);

  // Shuffle docs
  const shuffledDocs = [...docs];
  for (let i = shuffledDocs.length - 1; i > 0; i--) {
    const j = Math.floor(jsRng() * (i + 1));
    [shuffledDocs[i], shuffledDocs[j]] = [shuffledDocs[j], shuffledDocs[i]];
  }

  const tokenizer = buildTokenizer();

  // Init jax-js params
  const rngKey = random.key(seed);
  let params = await initParams(tokenizer.vocabSize, cfg, rngKey);

  // Count params
  const paramLeaves = tree.leaves(params);
  const paramCount = paramLeaves.reduce((s, p) => s + p.size, 0);

  // Optimizer — use schedule for LR
  const lrSchedule = (count) => baseLR * lrMultiplier(count, numSteps);
  const solver = adam(lrSchedule, { b1: 0.85, b2: 0.99 });
  let optState = solver.init(tree.ref(params));

  yield { type: 'init', vocabSize: tokenizer.vocabSize, paramCount, backend: backendName, model: modelName, cfg, batchSize };

  let docIdx = 0;
  for (let step = 0; step < numSteps; step++) {
    // Collect a batch of docs
    const batchTokens = [];
    for (let b = 0; b < batchSize; b++) {
      const doc = shuffledDocs[docIdx % shuffledDocs.length];
      docIdx++;
      const tokens = tokenizer.encode(doc);
      batchTokens.push(tokens);
    }

    // Pad to max sequence length in batch (capped at blockSize)
    const maxLen = Math.min(cfg.blockSize, Math.max(...batchTokens.map(t => t.length - 1)));

    // Build padded input/target arrays and mask [B, maxLen]
    const inputBuf = new Int32Array(batchSize * maxLen);
    const targetBuf = new Int32Array(batchSize * maxLen);
    const maskBuf = new Float32Array(batchSize * maxLen);

    for (let b = 0; b < batchSize; b++) {
      const tokens = batchTokens[b];
      const n = Math.min(maxLen, tokens.length - 1);
      for (let i = 0; i < n; i++) {
        inputBuf[b * maxLen + i] = tokens[i];
        targetBuf[b * maxLen + i] = tokens[i + 1];
        maskBuf[b * maxLen + i] = 1.0;
      }
      // Remaining positions stay 0 (padded), mask stays 0
    }

    const inputIds = np.array(inputBuf, { dtype: np.int32 }).reshape([batchSize, maxLen]);
    const posIds = np.tile(np.arange(maxLen).astype(np.int32), [batchSize, 1]);
    const targetIds = np.array(targetBuf, { dtype: np.int32 }).reshape([batchSize, maxLen]);
    const mask = np.array(maskBuf).reshape([batchSize, maxLen]);

    // Pre-compute oneHot matrices OUTSIDE valueAndGrad (avoids tracing issues)
    const tokenOH = nn.oneHot(inputIds, tokenizer.vocabSize);   // [B, maxLen, vocabSize]
    const posOH = nn.oneHot(posIds, cfg.blockSize);              // [B, maxLen, blockSize]
    const targetOH = nn.oneHot(targetIds, tokenizer.vocabSize);  // [B, maxLen, vocabSize]

    // Forward + backward (ref params since valueAndGrad consumes its arg)
    const lossFn = (p) => loss(p, cfg, tokenOH.ref, posOH.ref, targetOH.ref, maxLen, mask.ref, true);
    const [lossVal, grads] = valueAndGrad(lossFn)(tree.ref(params));

    // Read loss
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
    // mask is consumed by valueAndGrad via .ref — no manual dispose needed

    yield { type: 'step', step: step + 1, loss: lossScalar, n: maxLen, totalSteps: numSteps, batchSize };
  }

  // Inference
  const inferKey = random.key(seed + 1);
  const samples = await inference(params, cfg, tokenizer, inferKey, {
    temperature: 0.5,
    numSamples: 20,
  });

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

  const modelName = getArg('model', 'nano');
  const backendArg = getArg('backend', 'cpu');
  const steps = getArg('steps', null);
  const batch = getArg('batch', '8');

  // Initialize jax-js
  const devices = await init();
  const device = backendArg === 'webgpu' ? 'webgpu' : 'wasm';
  if (!devices.includes(device)) {
    console.error(`Backend "${device}" not available. Available: ${devices.join(', ')}`);
    if (typeof Deno !== 'undefined') Deno.exit(1); else process.exit(1);
  }
  defaultDevice(device);

  const text = isDeno
    ? Deno.readTextFileSync('input.txt')
    : (await import('fs')).readFileSync('input.txt', 'utf-8');
  const docs = text.split('\n').filter(l => l.trim());

  const backendDisplay = device === 'wasm' ? 'cpu (wasm)' : device;
  console.log(`model: ${modelName}`);
  console.log(`num docs: ${docs.length}`);
  console.log(`backend: ${backendDisplay}`);

  const trainOpts = { model: modelName, batchSize: parseInt(batch) };
  if (steps) trainOpts.steps = parseInt(steps);

  const gen = train(device, docs, trainOpts);

  const write = isDeno
    ? (s) => Deno.stdout.writeSync(new TextEncoder().encode(s))
    : (s) => process.stdout.write(s);

  for await (const event of gen) {
    if (event.type === 'init') {
      console.log(`vocab size: ${event.vocabSize}`);
      console.log(`batch size: ${event.batchSize}`);
      console.log(`num params: ${event.paramCount.toLocaleString()}`);
    } else if (event.type === 'step') {
      write(`\rstep ${String(event.step).padStart(4)} / ${String(event.totalSteps).padStart(4)} | loss ${event.loss.toFixed(4)}`);
    } else if (event.type === 'done') {
      console.log('\n--- inference (new, hallucinated names) ---');
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
