// quectoGPT training — training loop with jax-js
// Node: node train.js [--backend=cpu|webgpu] [--steps=200] [--model=medium]
// Deno: deno run --allow-read --allow-net --unstable-webgpu train.js --backend=webgpu --model=medium
// Browser: imported by index.html

import { init, defaultDevice, numpy as np, nn, valueAndGrad, random, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';
import { MODEL_CONFIGS, resolveConfig, initParams, loss, inference } from './gpt.js';

export { MODEL_CONFIGS };

// --- Tokenizer ---
export function buildTokenizer(docs) {
  const uchars = [...new Set(docs.join(''))].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;
  const encode = (str) => [BOS, ...str.split('').map(c => uchars.indexOf(c)), BOS];
  const decode = (ids) => ids.filter(id => id !== BOS).map(id => uchars[id]).join('');
  return { uchars, BOS, vocabSize, encode, decode };
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
  const baseLR = opts.lr ?? cfg.lr;
  const seed = opts.seed ?? 42;
  const jsRng = mulberry32(seed);

  // Shuffle docs
  const shuffledDocs = [...docs];
  for (let i = shuffledDocs.length - 1; i > 0; i--) {
    const j = Math.floor(jsRng() * (i + 1));
    [shuffledDocs[i], shuffledDocs[j]] = [shuffledDocs[j], shuffledDocs[i]];
  }

  const tokenizer = buildTokenizer(shuffledDocs);

  // Init jax-js params
  const rngKey = random.key(seed);
  let params = initParams(tokenizer.vocabSize, cfg, rngKey);

  // Count params
  const paramLeaves = tree.leaves(params);
  const paramCount = paramLeaves.reduce((s, p) => s + p.size, 0);

  // Optimizer — use schedule for LR
  const lrSchedule = (count) => baseLR * lrMultiplier(count, numSteps);
  const solver = adam(lrSchedule, { b1: 0.85, b2: 0.99 });
  let optState = solver.init(tree.ref(params));

  yield { type: 'init', vocabSize: tokenizer.vocabSize, paramCount, backend: backendName, model: modelName, cfg };

  for (let step = 0; step < numSteps; step++) {
    const doc = shuffledDocs[step % shuffledDocs.length];
    const tokens = tokenizer.encode(doc);
    const n = Math.min(cfg.blockSize, tokens.length - 1);

    const inputIds = np.array(new Int32Array(tokens.slice(0, n)), { dtype: np.int32 });
    const posIds = np.arange(n).astype(np.int32);
    const targetIds = np.array(new Int32Array(tokens.slice(1, n + 1)), { dtype: np.int32 });

    // Pre-compute oneHot matrices OUTSIDE valueAndGrad (avoids tracing issues)
    const tokenOH = nn.oneHot(inputIds, tokenizer.vocabSize);
    const posOH = nn.oneHot(posIds, cfg.blockSize);
    const targetOH = nn.oneHot(targetIds, tokenizer.vocabSize);

    // Forward + backward (ref params since valueAndGrad consumes its arg)
    const lossFn = (p) => loss(p, cfg, tokenOH.ref, posOH.ref, targetOH.ref, n);
    const [lossVal, grads] = valueAndGrad(lossFn)(tree.ref(params));

    // Optimizer step
    const [updates, newOptState] = solver.update(grads, optState, tree.ref(params));
    params = applyUpdates(params, updates);
    optState = newOptState;

    // Read loss
    const lossScalar = lossVal.item();

    yield { type: 'step', step: step + 1, loss: lossScalar, n, totalSteps: numSteps };
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

  const trainOpts = { model: modelName };
  if (steps) trainOpts.steps = parseInt(steps);

  const gen = train(device, docs, trainOpts);

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

// Run if executed directly
if (typeof Deno !== 'undefined') {
  if (import.meta.main) main().catch(console.error);
} else if (typeof process !== 'undefined' && process.versions?.node) {
  const url = await import('url');
  if (import.meta.url === url.pathToFileURL(process.argv[1]).href) {
    main().catch(console.error);
  }
}
