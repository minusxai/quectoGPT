// quectoGPT benchmark + correctness tests
// node bench.js [--cpu] [--gpu] [--all]

import { Tensor, backward, zeroGrad, clearTape, ops } from './quectrograd/index.js';
import { initCPU } from './quectrograd/backend_cpu.js';

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

function timeIt(fn, warmup = 2, runs = 5) {
  for (let i = 0; i < warmup; i++) fn();
  const times = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    times.push(performance.now() - t0);
  }
  return times.reduce((a, b) => a + b) / times.length;
}

function pass(name) { console.log(`  ✓ ${name}`); }
function fail(name, detail) { console.log(`  ✗ ${name}: ${detail}`); }

// --- Gradient checking via finite differences ---
async function finiteDiffCheck(opFn, inputShapes, backend, eps = 1e-4) {
  const inputs = inputShapes.map(shape => {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = (Math.random() - 0.5) * 2;
    return Tensor.from(data, shape, backend, { requiresGrad: true });
  });

  // Forward + backward with autograd
  clearTape();
  zeroGrad(inputs);
  const out = opFn(...inputs);
  const sumOut = ops.scale(out, 1.0); // identity to get on tape
  backward(sumOut);

  const results = [];
  for (let inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
    const input = inputs[inputIdx];
    const analyticGrad = input.grad ? await input.grad.toArray() : new Float32Array(input.numel());
    const numericGrad = new Float32Array(input.numel());
    const inputData = await input.toArray();

    for (let i = 0; i < input.numel(); i++) {
      const plusData = new Float32Array(inputData);
      plusData[i] += eps;
      const plusInputs = inputs.map((inp, idx) =>
        idx === inputIdx ? Tensor.from(plusData, inp.shape, backend, { requiresGrad: true }) : inp
      );
      clearTape();
      const plusOut = opFn(...plusInputs);
      const plusArr = await plusOut.toArray();
      let plusSum = 0;
      for (let j = 0; j < plusArr.length; j++) plusSum += plusArr[j];

      const minusData = new Float32Array(inputData);
      minusData[i] -= eps;
      const minusInputs = inputs.map((inp, idx) =>
        idx === inputIdx ? Tensor.from(minusData, inp.shape, backend, { requiresGrad: true }) : inp
      );
      clearTape();
      const minusOut = opFn(...minusInputs);
      const minusArr = await minusOut.toArray();
      let minusSum = 0;
      for (let j = 0; j < minusArr.length; j++) minusSum += minusArr[j];

      numericGrad[i] = (plusSum - minusSum) / (2 * eps);
    }

    results.push({ inputIdx, err: relError(analyticGrad, numericGrad) });
  }
  return results;
}

// --- Correctness Tests ---
async function runCorrectnessTests(backend) {
  console.log('\n=== Correctness Tests ===\n');

  {
    const a = Tensor.from(new Float32Array([1,2,3,4,5,6]), [2, 3], backend);
    const b = Tensor.from(new Float32Array([7,8,9,10,11,12]), [3, 2], backend);
    clearTape();
    const c = ops.matmul(a, b);
    const result = await c.toArray();
    const expected = new Float32Array([58, 64, 139, 154]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`matmul forward (err=${err.toExponential(2)})`);
    else fail(`matmul forward`, `err=${err}`);
  }

  {
    const x = Tensor.from(new Float32Array([1, 2]), [1, 2], backend);
    const w = Tensor.from(new Float32Array([3, 4, 5, 6, 7, 8]), [3, 2], backend);
    clearTape();
    const out = ops.matmulWT(x, w);
    const result = await out.toArray();
    const expected = new Float32Array([11, 17, 23]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`matmulWT forward (err=${err.toExponential(2)})`);
    else fail(`matmulWT forward`, `err=${err}`);
  }

  {
    const a = Tensor.from(new Float32Array([1, 2, 3]), [1, 3], backend);
    const b = Tensor.from(new Float32Array([4, 5, 6]), [1, 3], backend);
    clearTape();
    const c = ops.add(a, b);
    const result = await c.toArray();
    const expected = new Float32Array([5, 7, 9]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`add forward (err=${err.toExponential(2)})`);
    else fail(`add forward`, `err=${err}`);
  }

  {
    const x = Tensor.from(new Float32Array([-2, -1, 0, 1, 2]), [1, 5], backend);
    clearTape();
    const y = ops.relu(x);
    const result = await y.toArray();
    const expected = new Float32Array([0, 0, 0, 1, 2]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`relu forward (err=${err.toExponential(2)})`);
    else fail(`relu forward`, `err=${err}`);
  }

  {
    const x = Tensor.from(new Float32Array([1, 2, 3, 1, 2, 3]), [2, 3], backend);
    clearTape();
    const y = ops.softmax(x);
    const result = await y.toArray();
    const sum0 = result[0] + result[1] + result[2];
    const sum1 = result[3] + result[4] + result[5];
    if (Math.abs(sum0 - 1) < 1e-5 && Math.abs(sum1 - 1) < 1e-5) pass('softmax forward (rows sum to 1)');
    else fail('softmax forward', `sums: ${sum0}, ${sum1}`);
  }

  {
    const x = Tensor.from(new Float32Array([3, 4]), [1, 2], backend);
    clearTape();
    const y = ops.rmsnorm(x);
    const result = await y.toArray();
    const ms = (9 + 16) / 2;
    const sc = 1.0 / Math.sqrt(ms + 1e-5);
    const expected = new Float32Array([3 * sc, 4 * sc]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`rmsnorm forward (err=${err.toExponential(2)})`);
    else fail(`rmsnorm forward`, `err=${err}`);
  }

  {
    const table = Tensor.from(new Float32Array([10, 20, 30, 40, 50, 60]), [3, 2], backend);
    const ids = new Int32Array([2, 0, 1]);
    clearTape();
    const y = ops.embedding(table, ids);
    const result = await y.toArray();
    const expected = new Float32Array([50, 60, 10, 20, 30, 40]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`embedding forward (err=${err.toExponential(2)})`);
    else fail(`embedding forward`, `err=${err}`);
  }

  {
    const logits = Tensor.from(new Float32Array([2, 1, 0.1]), [1, 3], backend, { requiresGrad: true });
    const targets = new Int32Array([0]);
    clearTape();
    const loss = ops.crossEntropyLoss(logits, targets);
    const lossVal = (await loss.toArray())[0];
    const exps = [Math.exp(2), Math.exp(1), Math.exp(0.1)];
    const total = exps[0] + exps[1] + exps[2];
    const expected = -Math.log(exps[0] / total);
    const err = Math.abs(lossVal - expected);
    if (err < 1e-4) pass(`crossEntropyLoss forward (err=${err.toExponential(2)})`);
    else fail(`crossEntropyLoss forward`, `got=${lossVal}, expected=${expected}, err=${err}`);
  }

  {
    const x = Tensor.from(new Float32Array([1,2,3,4, 5,6,7,8]), [2, 4], backend);
    clearTape();
    const s = ops.sliceCols(x, 1, 2);
    const result = await s.toArray();
    const expected = new Float32Array([2, 3, 6, 7]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`sliceCols forward (err=${err.toExponential(2)})`);
    else fail(`sliceCols forward`, `err=${err}`);
  }

  {
    const a = Tensor.from(new Float32Array([1, 2, 3, 4]), [2, 2], backend);
    clearTape();
    const t = ops.transpose2D(a);
    const result = await t.toArray();
    const expected = new Float32Array([1, 3, 2, 4]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`transpose2D forward (err=${err.toExponential(2)})`);
    else fail(`transpose2D forward`, `err=${err}`);
  }

  console.log('\n=== Gradient Checks ===\n');

  const gradChecks = [
    { name: 'matmul', fn: (a, b) => ops.matmul(a, b), shapes: [[2, 3], [3, 2]] },
    { name: 'matmulWT', fn: (x, w) => ops.matmulWT(x, w), shapes: [[2, 3], [4, 3]] },
    { name: 'add', fn: (a, b) => ops.add(a, b), shapes: [[2, 3], [2, 3]] },
    { name: 'relu', fn: (a) => ops.relu(a), shapes: [[2, 3]] },
    { name: 'scale', fn: (a) => ops.scale(a, 0.5), shapes: [[2, 3]] },
    { name: 'negate', fn: (a) => ops.negate(a), shapes: [[2, 3]] },
    { name: 'rmsnorm', fn: (a) => ops.rmsnorm(a), shapes: [[2, 4]] },
    { name: 'softmax', fn: (a) => ops.softmax(a), shapes: [[2, 4]] },
  ];

  for (const { name, fn, shapes } of gradChecks) {
    const results = await finiteDiffCheck(fn, shapes, backend);
    const maxErr = Math.max(...results.map(r => r.err));
    if (maxErr < 0.05) pass(`${name} gradient (max rel err=${maxErr.toExponential(2)})`);
    else fail(`${name} gradient`, `max rel err=${maxErr.toExponential(2)}`);
  }
}

// --- Op-level benchmarks ---
function runBenchmarks(backend) {
  console.log(`\n=== Op-level Benchmarks (${backend.name}) ===\n`);
  console.log('op             | size      | ms');
  console.log('---------------|-----------|-------');

  for (const n of [16, 64, 256]) {
    const a = Tensor.from(new Float32Array(n * n).fill(1), [n, n], backend);
    const b = Tensor.from(new Float32Array(n * n).fill(1), [n, n], backend);
    const ms = timeIt(() => { clearTape(); ops.matmul(a, b); });
    console.log(`matmul         | ${String(n + 'x' + n).padEnd(9)} | ${ms.toFixed(2)}`);
  }

  for (const n of [1000, 10000, 100000]) {
    const a = Tensor.from(new Float32Array(n).fill(1), [1, n], backend);
    const b = Tensor.from(new Float32Array(n).fill(1), [1, n], backend);
    const ms = timeIt(() => { clearTape(); ops.add(a, b); });
    console.log(`add            | ${String(n).padEnd(9)} | ${ms.toFixed(2)}`);
  }

  for (const n of [1000, 10000, 100000]) {
    const a = Tensor.from(new Float32Array(n).fill(1), [1, n], backend);
    const ms = timeIt(() => { clearTape(); ops.relu(a); });
    console.log(`relu           | ${String(n).padEnd(9)} | ${ms.toFixed(2)}`);
  }

  for (const n of [16, 64, 256]) {
    const a = Tensor.from(new Float32Array(n).fill(1), [1, n], backend);
    const ms = timeIt(() => { clearTape(); ops.softmax(a); });
    console.log(`softmax        | ${String(n).padEnd(9)} | ${ms.toFixed(2)}`);
  }
}

// --- End-to-end training benchmark ---
async function runTrainingBench(backend, docs, steps = 10) {
  console.log(`\n=== Training Benchmark (${backend.name}, ${steps} steps) ===\n`);
  const { train } = await import('./train.js');

  const t0 = performance.now();
  const losses = [];
  const gen = train(backend, docs, { steps });

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

// --- Main ---
async function main() {
  const args = process.argv.slice(2);
  const runCPU = args.includes('--cpu') || args.includes('--all') || args.length === 0;
  const runGPU = args.includes('--gpu') || args.includes('--all');

  const cpuBackend = initCPU();

  if (runCPU) {
    await runCorrectnessTests(cpuBackend);
    runBenchmarks(cpuBackend);

    const fs = await import('fs');
    const text = fs.readFileSync('input.txt', 'utf-8');
    const docs = text.split('\n').filter(l => l.trim());
    await runTrainingBench(cpuBackend, docs, 10);
  }

  if (runGPU) {
    try {
      const { initWebGPU } = await import('./quectrograd/backend_webgpu.js');
      const gpuBackend = await initWebGPU();
      console.log('\n=== WebGPU Backend Available ===');
      await runCorrectnessTests(gpuBackend);
      runBenchmarks(gpuBackend);

      const fs = await import('fs');
      const text = fs.readFileSync('input.txt', 'utf-8');
      const docs = text.split('\n').filter(l => l.trim());
      await runTrainingBench(gpuBackend, docs, 10);
    } catch (e) {
      console.log(`\nWebGPU not available: ${e.message}`);
    }
  }
}

main().catch(console.error);
