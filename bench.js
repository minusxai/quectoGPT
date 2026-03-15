// quectoGPT benchmark + correctness tests
// node bench.js [--cpu] [--gpu] [--all]

import { Tensor, backward, zeroGrad, clearTape, ops, Adam } from './quectrograd/index.js';
import { initCPU } from './quectrograd/backend_cpu.js';

// --- Utilities ---
function maxAbsError(a, b) {
  let maxErr = 0;
  for (let i = 0; i < a.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(a[i] - b[i]));
  }
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
function finiteDiffCheck(opFn, inputShapes, backend, eps = 1e-4, rtol = 1e-2) {
  // Create random inputs
  const inputs = inputShapes.map(shape => {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = (Math.random() - 0.5) * 2;
    return Tensor.from(data, shape, backend, { requiresGrad: true });
  });

  // Forward + backward
  clearTape();
  const out = opFn(...inputs);
  // Sum all outputs for scalar loss
  const outArr = out.toArray();
  let totalLoss = 0;
  for (let i = 0; i < outArr.length; i++) totalLoss += outArr[i];

  // Create a fake scalar sum tensor
  clearTape();
  const out2 = opFn(...inputs);
  // Manually set grad to all ones (upstream gradient = 1 for each element)
  out2.ensureGrad();
  backend.fill(out2.grad.handle, 1.0, out2.numel());
  // Walk tape backward
  for (let i = clearTape.length - 1; i >= 0; i--) {
    // Can't do this easily... let's use the proper backward
  }

  // Use proper autograd
  clearTape();
  zeroGrad(inputs);
  const out3 = opFn(...inputs);

  // To get a scalar, sum the output
  const sumOut = ops.scale(out3, 1.0); // identity, just to have something on tape
  backward(sumOut);

  // Actually, backward seeds with grad=1 on sumOut which is the same shape as out3.
  // We want sum(out3). Let's compute finite diff element by element.

  // Simpler approach: for each input element, perturb and measure change in sum(output)
  const results = [];
  for (let inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
    const input = inputs[inputIdx];
    const analyticGrad = input.grad ? input.grad.toArray() : new Float32Array(input.numel());
    const numericGrad = new Float32Array(input.numel());
    const inputData = input.toArray();

    for (let i = 0; i < input.numel(); i++) {
      // f(x + eps)
      const plusData = new Float32Array(inputData);
      plusData[i] += eps;
      const plusInputs = inputs.map((inp, idx) =>
        idx === inputIdx
          ? Tensor.from(plusData, inp.shape, backend, { requiresGrad: true })
          : inp
      );
      clearTape();
      const plusOut = opFn(...plusInputs);
      const plusArr = plusOut.toArray();
      let plusSum = 0;
      for (let j = 0; j < plusArr.length; j++) plusSum += plusArr[j];

      // f(x - eps)
      const minusData = new Float32Array(inputData);
      minusData[i] -= eps;
      const minusInputs = inputs.map((inp, idx) =>
        idx === inputIdx
          ? Tensor.from(minusData, inp.shape, backend, { requiresGrad: true })
          : inp
      );
      clearTape();
      const minusOut = opFn(...minusInputs);
      const minusArr = minusOut.toArray();
      let minusSum = 0;
      for (let j = 0; j < minusArr.length; j++) minusSum += minusArr[j];

      numericGrad[i] = (plusSum - minusSum) / (2 * eps);
    }

    const err = relError(analyticGrad, numericGrad);
    results.push({ inputIdx, err, analyticGrad, numericGrad });
  }

  return results;
}

// --- Correctness Tests ---
function runCorrectnessTests(backend) {
  console.log('\n=== Correctness Tests (CPU) ===\n');

  // Test matmul forward
  {
    const a = Tensor.from(new Float32Array([1,2,3,4,5,6]), [2, 3], backend, { requiresGrad: true });
    const b = Tensor.from(new Float32Array([7,8,9,10,11,12]), [3, 2], backend, { requiresGrad: true });
    clearTape();
    const c = ops.matmul(a, b);
    const result = c.toArray();
    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //         = [[58, 64], [139, 154]]
    const expected = new Float32Array([58, 64, 139, 154]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`matmul forward (err=${err.toExponential(2)})`);
    else fail(`matmul forward`, `err=${err}`);
  }

  // Test matmulWT forward
  {
    // W is [3, 2] (outDim=3, inDim=2), X is [1, 2]
    // out[i,j] = sum_k x[i,k] * w[j,k]
    const x = Tensor.from(new Float32Array([1, 2]), [1, 2], backend, { requiresGrad: true });
    const w = Tensor.from(new Float32Array([3, 4, 5, 6, 7, 8]), [3, 2], backend, { requiresGrad: true });
    clearTape();
    const out = ops.matmulWT(x, w);
    const result = out.toArray();
    // out[0,0] = 1*3 + 2*4 = 11
    // out[0,1] = 1*5 + 2*6 = 17
    // out[0,2] = 1*7 + 2*8 = 23
    const expected = new Float32Array([11, 17, 23]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`matmulWT forward (err=${err.toExponential(2)})`);
    else fail(`matmulWT forward`, `err=${err}`);
  }

  // Test add forward
  {
    const a = Tensor.from(new Float32Array([1, 2, 3]), [1, 3], backend, { requiresGrad: true });
    const b = Tensor.from(new Float32Array([4, 5, 6]), [1, 3], backend, { requiresGrad: true });
    clearTape();
    const c = ops.add(a, b);
    const result = c.toArray();
    const expected = new Float32Array([5, 7, 9]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`add forward (err=${err.toExponential(2)})`);
    else fail(`add forward`, `err=${err}`);
  }

  // Test relu forward
  {
    const x = Tensor.from(new Float32Array([-2, -1, 0, 1, 2]), [1, 5], backend, { requiresGrad: true });
    clearTape();
    const y = ops.relu(x);
    const result = y.toArray();
    const expected = new Float32Array([0, 0, 0, 1, 2]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`relu forward (err=${err.toExponential(2)})`);
    else fail(`relu forward`, `err=${err}`);
  }

  // Test softmax forward
  {
    const x = Tensor.from(new Float32Array([1, 2, 3, 1, 2, 3]), [2, 3], backend);
    clearTape();
    const y = ops.softmax(x);
    const result = y.toArray();
    // Each row should sum to 1
    const sum0 = result[0] + result[1] + result[2];
    const sum1 = result[3] + result[4] + result[5];
    if (Math.abs(sum0 - 1) < 1e-5 && Math.abs(sum1 - 1) < 1e-5) pass('softmax forward (rows sum to 1)');
    else fail('softmax forward', `sums: ${sum0}, ${sum1}`);
  }

  // Test rmsnorm forward
  {
    const x = Tensor.from(new Float32Array([3, 4]), [1, 2], backend);
    clearTape();
    const y = ops.rmsnorm(x);
    const result = y.toArray();
    // rms = sqrt((9+16)/2) = sqrt(12.5), scale = 1/sqrt(12.5 + 1e-5)
    const ms = (9 + 16) / 2;
    const sc = 1.0 / Math.sqrt(ms + 1e-5);
    const expected = new Float32Array([3 * sc, 4 * sc]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`rmsnorm forward (err=${err.toExponential(2)})`);
    else fail(`rmsnorm forward`, `err=${err}`);
  }

  // Test embedding forward
  {
    const table = Tensor.from(new Float32Array([10, 20, 30, 40, 50, 60]), [3, 2], backend, { requiresGrad: true });
    const ids = new Int32Array([2, 0, 1]);
    clearTape();
    const y = ops.embedding(table, ids);
    const result = y.toArray();
    const expected = new Float32Array([50, 60, 10, 20, 30, 40]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`embedding forward (err=${err.toExponential(2)})`);
    else fail(`embedding forward`, `err=${err}`);
  }

  // Test cross-entropy loss
  {
    const logits = Tensor.from(new Float32Array([2, 1, 0.1]), [1, 3], backend, { requiresGrad: true });
    const targets = new Int32Array([0]);
    clearTape();
    const loss = ops.crossEntropyLoss(logits, targets);
    const lossVal = loss.toArray()[0];
    // Manual: softmax([2,1,0.1]) then -log(p[0])
    const exps = [Math.exp(2), Math.exp(1), Math.exp(0.1)];
    const total = exps[0] + exps[1] + exps[2];
    const expected = -Math.log(exps[0] / total);
    const err = Math.abs(lossVal - expected);
    if (err < 1e-4) pass(`crossEntropyLoss forward (err=${err.toExponential(2)})`);
    else fail(`crossEntropyLoss forward`, `got=${lossVal}, expected=${expected}, err=${err}`);
  }

  // Test sliceCols / concatCols / stackRows / transpose2D
  {
    const x = Tensor.from(new Float32Array([1,2,3,4, 5,6,7,8]), [2, 4], backend);
    clearTape();
    const s = ops.sliceCols(x, 1, 2);
    const result = s.toArray();
    const expected = new Float32Array([2, 3, 6, 7]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`sliceCols forward (err=${err.toExponential(2)})`);
    else fail(`sliceCols forward`, `err=${err}`);
  }

  {
    const a = Tensor.from(new Float32Array([1, 2, 3, 4]), [2, 2], backend);
    clearTape();
    const t = ops.transpose2D(a);
    const result = t.toArray();
    const expected = new Float32Array([1, 3, 2, 4]);
    const err = maxAbsError(result, expected);
    if (err < 1e-5) pass(`transpose2D forward (err=${err.toExponential(2)})`);
    else fail(`transpose2D forward`, `err=${err}`);
  }

  console.log('\n=== Gradient Checks ===\n');

  // Gradient checks via finite differences
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
    const results = finiteDiffCheck(fn, shapes, backend);
    const maxErr = Math.max(...results.map(r => r.err));
    if (maxErr < 0.05) pass(`${name} gradient (max rel err=${maxErr.toExponential(2)})`);
    else fail(`${name} gradient`, `max rel err=${maxErr.toExponential(2)}`);
  }
}

// --- Op-level benchmarks ---
function runBenchmarks(backend) {
  console.log('\n=== Op-level Benchmarks (CPU) ===\n');
  console.log('op             | size      | ms');
  console.log('---------------|-----------|-------');

  const sizes = [16, 64, 256];

  for (const n of sizes) {
    const a = Tensor.from(new Float32Array(n * n).fill(1), [n, n], backend);
    const b = Tensor.from(new Float32Array(n * n).fill(1), [n, n], backend);
    const ms = timeIt(() => {
      clearTape();
      ops.matmul(a, b);
    });
    console.log(`matmul         | ${String(n + 'x' + n).padEnd(9)} | ${ms.toFixed(2)}`);
  }

  const elSizes = [1000, 10000, 100000];
  for (const n of elSizes) {
    const a = Tensor.from(new Float32Array(n).fill(1), [1, n], backend);
    const b = Tensor.from(new Float32Array(n).fill(1), [1, n], backend);
    const ms = timeIt(() => { clearTape(); ops.add(a, b); });
    console.log(`add            | ${String(n).padEnd(9)} | ${ms.toFixed(2)}`);
  }

  for (const n of elSizes) {
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
    if (event.type === 'step') {
      losses.push(event.loss);
    }
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
    runCorrectnessTests(cpuBackend);
    runBenchmarks(cpuBackend);

    // Training bench
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

      // Run same benchmarks on GPU
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
