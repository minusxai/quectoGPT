// quectoGPT benchmark + correctness tests using jax-js
// node bench.js [--cpu] [--gpu] [--all]

import { init, defaultDevice, numpy as np, nn, grad, random, tree, blockUntilReady } from '@jax-js/jax';

// --- Utilities ---
function maxAbsError(a, b) {
  let maxErr = 0;
  for (let i = 0; i < a.length; i++) maxErr = Math.max(maxErr, Math.abs(a[i] - b[i]));
  return maxErr;
}

function relError(a, b) {
  let maxRel = 0;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(Math.abs(a[i]), Math.abs(b[i]), 1e-8);
    maxRel = Math.max(maxRel, Math.abs(a[i] - b[i]) / denom);
  }
  return maxRel;
}

async function timeIt(fn, warmup = 10, runs = 100) {
  for (let i = 0; i < warmup; i++) await fn();
  const times = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }
  return times.reduce((a, b) => a + b) / times.length;
}

function pass(name) { console.log(`  ✓ ${name}`); }
function fail(name, detail) { console.log(`  ✗ ${name}: ${detail}`); }

// --- Gradient checking via finite differences ---
async function finiteDiffCheck(opFn, inputShapes, eps = 1e-4) {
  const rngKey = random.key(123);
  const keys = random.split(rngKey, inputShapes.length);

  const inputs = inputShapes.map((shape, idx) => {
    const k = keys.ref.slice(idx);
    return random.uniform(k, shape, { minval: -1, maxval: 1 });
  });

  // Read concrete data for finite differences
  const inputDatas = inputs.map(inp => inp.ref.dataSync());
  const inputShapesActual = inputs.map(inp => inp.ref.shape);

  const sumFn = (...args) => np.sum(opFn(...args));
  const results = [];

  for (let inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
    // Analytic gradient via grad()
    const gradFn = grad(sumFn, { argnums: inputIdx });
    const analyticGrad = gradFn(...inputs.map(x => x.ref));
    const analyticArr = analyticGrad.dataSync();

    // Numeric gradient via finite differences
    const inputData = inputDatas[inputIdx];
    const shape = inputShapesActual[inputIdx];
    const numericGrad = new Float32Array(inputData.length);

    for (let i = 0; i < inputData.length; i++) {
      const plusData = new Float32Array(inputData);
      plusData[i] += eps;
      const plusInputs = inputs.map((inp, idx) =>
        idx === inputIdx ? np.array(plusData).reshape(shape) : inp.ref
      );
      const plusVal = np.sum(opFn(...plusInputs)).item();

      const minusData = new Float32Array(inputData);
      minusData[i] -= eps;
      const minusInputs = inputs.map((inp, idx) =>
        idx === inputIdx ? np.array(minusData).reshape(shape) : inp.ref
      );
      const minusVal = np.sum(opFn(...minusInputs)).item();

      numericGrad[i] = (plusVal - minusVal) / (2 * eps);
    }

    results.push({ inputIdx, err: relError(analyticArr, numericGrad) });
  }

  tree.dispose(inputs);
  return results;
}

// --- Correctness Tests ---
async function runCorrectnessTests() {
  console.log('\n=== Correctness Tests ===\n');

  {
    const a = np.array([[1,2,3],[4,5,6]]);
    const b = np.array([[7,8],[9,10],[11,12]]);
    const c = np.matmul(a, b);
    const result = c.dataSync();
    const expected = new Float32Array([58, 64, 139, 154]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`matmul forward (err=${err.toExponential(2)})`);
    else fail(`matmul forward`, `err=${err}`);
  }

  {
    const x = np.array([[1, 2]]);
    const w = np.array([[3, 5, 7], [4, 6, 8]]);  // [inDim=2, outDim=3]
    const out = np.dot(x, w);
    const result = out.dataSync();
    const expected = new Float32Array([11, 17, 23]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`dot (linear) forward (err=${err.toExponential(2)})`);
    else fail(`dot (linear) forward`, `err=${err}`);
  }

  {
    const a = np.array([[1, 2, 3]]);
    const b = np.array([[4, 5, 6]]);
    const c = a.add(b);
    const result = c.dataSync();
    const expected = new Float32Array([5, 7, 9]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`add forward (err=${err.toExponential(2)})`);
    else fail(`add forward`, `err=${err}`);
  }

  {
    const x = np.array([[-2, -1, 0, 1, 2]]);
    const y = nn.relu(x);
    const result = y.dataSync();
    const expected = new Float32Array([0, 0, 0, 1, 2]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`relu forward (err=${err.toExponential(2)})`);
    else fail(`relu forward`, `err=${err}`);
  }

  {
    const x = np.array([[1, 2, 3], [1, 2, 3]]);
    const y = nn.softmax(x, -1);
    const result = y.dataSync();
    const sum0 = result[0] + result[1] + result[2];
    const sum1 = result[3] + result[4] + result[5];
    if (Math.abs(sum0 - 1) < 1e-5 && Math.abs(sum1 - 1) < 1e-5) pass('softmax forward (rows sum to 1)');
    else fail('softmax forward', `sums: ${sum0}, ${sum1}`);
  }

  {
    const x = np.array([[3, 4]]);
    const ms = (9 + 16) / 2;
    const sc = 1.0 / Math.sqrt(ms + 1e-5);
    const xRef = x.ref;
    const msArr = np.mean(np.square(xRef), -1, { keepdims: true });
    const y = x.div(np.sqrt(msArr.add(1e-5)));
    const result = y.dataSync();
    const expected = new Float32Array([3 * sc, 4 * sc]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`rmsnorm forward (err=${err.toExponential(2)})`);
    else fail(`rmsnorm forward`, `err=${err}`);
  }

  {
    const table = np.array([[10, 20], [30, 40], [50, 60]]);
    const ids = np.array(new Int32Array([2, 0, 1]), { dtype: np.int32 });
    const y = np.take(table, ids, 0);
    const result = y.dataSync();
    const expected = new Float32Array([50, 60, 10, 20, 30, 40]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`embedding (take) forward (err=${err.toExponential(2)})`);
    else fail(`embedding (take) forward`, `err=${err}`);
  }

  {
    const logits = np.array([[2.0, 1.0, 0.1]]);
    const targets = np.array(new Int32Array([0]), { dtype: np.int32 });
    const logprobs = nn.logSoftmax(logits, -1);
    const oneHot = nn.oneHot(targets, 3);
    const lossVal = np.mean(np.sum(logprobs.mul(oneHot), -1)).neg();
    const lossScalar = lossVal.item();
    const exps = [Math.exp(2), Math.exp(1), Math.exp(0.1)];
    const total = exps[0] + exps[1] + exps[2];
    const expected = -Math.log(exps[0] / total);
    const err = Math.abs(lossScalar - expected);
    if (err < 1e-4) pass(`crossEntropyLoss forward (err=${err.toExponential(2)})`);
    else fail(`crossEntropyLoss forward`, `got=${lossScalar}, expected=${expected}, err=${err}`);
  }

  {
    const a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]);
    const s = a.slice([], [1, 3]);  // a[:, 1:3]
    const result = s.dataSync();
    const expected = new Float32Array([2, 3, 6, 7]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`slice forward (err=${err.toExponential(2)})`);
    else fail(`slice forward`, `err=${err}`);
  }

  {
    const a = np.array([[1, 2], [3, 4]]);
    const t = np.transpose(a);
    const result = t.dataSync();
    const expected = new Float32Array([1, 3, 2, 4]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`transpose forward (err=${err.toExponential(2)})`);
    else fail(`transpose forward`, `err=${err}`);
  }

  console.log('\n=== Gradient Checks ===\n');

  const gradChecks = [
    { name: 'matmul', fn: (a, b) => np.matmul(a, b), shapes: [[2, 3], [3, 2]] },
    { name: 'dot (linear)', fn: (x, w) => np.dot(x, w), shapes: [[2, 3], [3, 4]] },
    { name: 'add', fn: (a, b) => a.add(b), shapes: [[2, 3], [2, 3]] },
    { name: 'relu', fn: (a) => nn.relu(a), shapes: [[2, 3]] },
    { name: 'scale', fn: (a) => a.mul(0.5), shapes: [[2, 3]] },
    { name: 'negate', fn: (a) => a.neg(), shapes: [[2, 3]] },
    { name: 'rmsnorm', fn: (a) => {
      const ms = np.mean(np.square(a.ref), -1, { keepdims: true });
      return a.div(np.sqrt(ms.add(1e-5)));
    }, shapes: [[2, 4]] },
    { name: 'softmax', fn: (a) => nn.softmax(a, -1), shapes: [[2, 4]] },
  ];

  for (const { name, fn, shapes } of gradChecks) {
    const results = await finiteDiffCheck(fn, shapes);
    const maxErr = Math.max(...results.map(r => r.err));
    if (maxErr < 0.05) pass(`${name} gradient (max rel err=${maxErr.toExponential(2)})`);
    else fail(`${name} gradient`, `max rel err=${maxErr.toExponential(2)}`);
  }
}

// --- Op-level benchmarks ---
async function runBenchmarks() {
  const device = defaultDevice();
  console.log(`\n=== Op-level Benchmarks (${device}) ===\n`);
  console.log('op             | size      | ms');
  console.log('---------------|-----------|-------');

  for (const n of [16, 64, 256]) {
    const a = np.ones([n, n]);
    const b = np.ones([n, n]);
    const ms = await timeIt(() => { const c = np.dot(a.ref, b.ref); c.dataSync(); });
    console.log(`matmul         | ${String(n + 'x' + n).padEnd(9)} | ${ms.toFixed(2)}`);
    a.dispose(); b.dispose();
  }

  for (const n of [1000, 10000, 100000]) {
    const a = np.ones([1, n]);
    const b = np.ones([1, n]);
    const ms = await timeIt(() => { const c = a.ref.add(b.ref); c.dataSync(); });
    console.log(`add            | ${String(n).padEnd(9)} | ${ms.toFixed(2)}`);
    a.dispose(); b.dispose();
  }

  for (const n of [1000, 10000, 100000]) {
    const a = np.ones([1, n]);
    const ms = await timeIt(() => { const c = nn.relu(a.ref); c.dataSync(); });
    console.log(`relu           | ${String(n).padEnd(9)} | ${ms.toFixed(2)}`);
    a.dispose();
  }

  for (const n of [16, 64, 256]) {
    const a = np.ones([1, n]);
    const ms = await timeIt(() => { const c = nn.softmax(a.ref, -1); c.dataSync(); });
    console.log(`softmax        | ${String(n).padEnd(9)} | ${ms.toFixed(2)}`);
    a.dispose();
  }
}

// --- End-to-end training benchmark ---
async function runTrainingBench(backendName, docs, steps = 10) {
  console.log(`\n=== Training Benchmark (${backendName}, ${steps} steps) ===\n`);
  const { train } = await import('./train.js');

  const t0 = performance.now();
  const losses = [];
  const gen = train(backendName, docs, { steps, model: 'nano' });

  for await (const event of gen) {
    if (event.type === 'step') losses.push(event.loss);
  }
  const elapsed = performance.now() - t0;

  console.log(`  elapsed: ${(elapsed / 1000).toFixed(2)}s`);
  console.log(`  steps/sec: ${(steps / (elapsed / 1000)).toFixed(2)}`);
  console.log(`  first loss: ${losses[0]?.toFixed(4)}`);
  console.log(`  last loss: ${losses[losses.length - 1]?.toFixed(4)}`);
  console.log(`  loss decreased: ${losses[losses.length - 1] < losses[0] ? 'yes' : 'NO'}`);
  return { elapsed, losses };
}

// --- File reading ---
async function readTextFile(path) {
  if (typeof Deno !== 'undefined') {
    return Deno.readTextFileSync(path);
  }
  const fs = await import('fs');
  return fs.readFileSync(path, 'utf-8');
}

// --- Main ---
async function main() {
  const args = typeof Deno !== 'undefined' ? Deno.args : process.argv.slice(2);
  const runCPU = args.includes('--cpu') || args.includes('--all') || args.length === 0;
  const runGPU = args.includes('--gpu') || args.includes('--all');

  if (runCPU) {
    const devices = await init();
    defaultDevice('wasm');

    await runCorrectnessTests();
    await runBenchmarks();

    const text = await readTextFile('input.txt');
    const docs = text.split('\n').filter(l => l.trim());
    await runTrainingBench('wasm', docs, 10);
  }

  if (runGPU) {
    try {
      const devices = await init();
      if (!devices.includes('webgpu')) throw new Error('WebGPU not available');
      defaultDevice('webgpu');
      console.log('\n=== WebGPU Backend Available ===');

      await runCorrectnessTests();
      await runBenchmarks();

      const text = await readTextFile('input.txt');
      const docs = text.split('\n').filter(l => l.trim());
      await runTrainingBench('webgpu', docs, 10);
    } catch (e) {
      console.log(`\nWebGPU not available: ${e.message}`);
    }
  }
}

// Export for browser use
export { runCorrectnessTests, runBenchmarks, runTrainingBench, finiteDiffCheck, maxAbsError, relError, timeIt };

main().catch(console.error);
