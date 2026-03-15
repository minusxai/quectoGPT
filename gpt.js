// quectoGPT model definition — pure functions, no side effects
// Uses jax-js for tensor operations + autodiff

import { numpy as np, nn, random, tree } from '@jax-js/jax';

// --- Model configs ---
export const MODEL_CONFIGS = {
  nano:   { nLayer: 1, nEmbd: 16,  blockSize: 16,  nHead: 4, lr: 0.01,  initStd: 0.08, steps: 100 },
  tiny:   { nLayer: 2, nEmbd: 64,  blockSize: 32,  nHead: 4, lr: 3e-3,  initStd: 0.04, steps: 200 },
  small:  { nLayer: 4, nEmbd: 128, blockSize: 64,  nHead: 4, lr: 1e-3,  initStd: 0.02, steps: 200 },
  medium: { nLayer: 6, nEmbd: 256, blockSize: 128, nHead: 8, lr: 3e-4,  initStd: 0.02, steps: 300 },
  large:  { nLayer: 8, nEmbd: 512, blockSize: 256, nHead: 8, lr: 1e-4,  initStd: 0.01, steps: 500 },
};

export function resolveConfig(name) {
  const cfg = MODEL_CONFIGS[name];
  if (!cfg) throw new Error(`Unknown model config: ${name}. Choose from: ${Object.keys(MODEL_CONFIGS).join(', ')}`);
  return { ...cfg, headDim: cfg.nEmbd / cfg.nHead };
}

// Helper: get the i-th key from a [N, 2] keys array
function getKey(keys, i) {
  return keys.ref.slice(i);
}

// --- Parameter initialization ---
export function initParams(vocabSize, cfg, rngKey) {
  const nEmbd = cfg.nEmbd;
  const s = Math.sqrt(3 / nEmbd);
  const numKeys = 3 + cfg.nLayer * 6;

  const keys = random.split(rngKey, numKeys);
  let ki = 0;
  const nextKey = () => getKey(keys, ki++);

  const params = {
    wte: random.normal(nextKey(), [vocabSize, nEmbd]).mul(cfg.initStd),
    wpe: random.normal(nextKey(), [cfg.blockSize, nEmbd]).mul(cfg.initStd),
    lmHead: random.normal(nextKey(), [vocabSize, nEmbd]).mul(0.001),
    layers: [],
  };

  for (let i = 0; i < cfg.nLayer; i++) {
    params.layers.push({
      wq: random.uniform(nextKey(), [nEmbd, nEmbd], { minval: -s, maxval: s }),
      wk: random.uniform(nextKey(), [nEmbd, nEmbd], { minval: -s, maxval: s }),
      wv: random.uniform(nextKey(), [nEmbd, nEmbd], { minval: -s, maxval: s }),
      wo: np.zeros([nEmbd, nEmbd]),
      mlpFc1: random.uniform(nextKey(), [4 * nEmbd, nEmbd], { minval: -0.4 * s, maxval: 0.4 * s }),
      mlpFc2: np.zeros([nEmbd, 4 * nEmbd]),
    });
  }

  keys.dispose();
  return params;
}

// --- RMSNorm (no learnable params) ---
function rmsnorm(x) {
  const ms = np.mean(np.square(x.ref), -1, { keepdims: true });
  return x.div(np.sqrt(ms.add(1e-5)));
}

// --- Linear: x @ w^T ---
function linear(x, w) {
  return np.matmul(x, np.transpose(w));
}

// --- Forward pass ---
export function forward(params, cfg, inputIds, posIds) {
  const headDim = cfg.nEmbd / cfg.nHead;

  // Token + position embeddings
  let x = np.take(params.wte, inputIds, 0);
  const posEmb = np.take(params.wpe, posIds, 0);
  x = rmsnorm(x.add(posEmb));

  for (let li = 0; li < cfg.nLayer; li++) {
    const layer = params.layers[li];
    const xResidual = x.ref;
    x = rmsnorm(x);

    // QKV projections: [seqLen, nEmbd] -> [seqLen, nEmbd]
    const q = linear(x.ref, layer.wq);
    const k = linear(x.ref, layer.wk);
    const v = linear(x, layer.wv);

    // Reshape to [seqLen, nHead, headDim] for dotProductAttention
    const seqLen = inputIds.shape[0];
    const qH = q.reshape([seqLen, cfg.nHead, headDim]);
    const kH = k.reshape([seqLen, cfg.nHead, headDim]);
    const vH = v.reshape([seqLen, cfg.nHead, headDim]);

    // Scaled dot-product attention with causal mask (rank-3: no batch dim)
    const attnOut = nn.dotProductAttention(qH, kH, vH, { isCausal: true });

    // Reshape back to [seqLen, nEmbd] and project
    const attnFlat = attnOut.reshape([seqLen, cfg.nEmbd]);
    x = linear(attnFlat, layer.wo).add(xResidual);

    // MLP block
    const mlpResidual = x.ref;
    x = rmsnorm(x);
    x = nn.relu(linear(x, layer.mlpFc1));
    x = linear(x, layer.mlpFc2).add(mlpResidual);
  }

  // Output logits: [seqLen, vocabSize]
  return linear(x, params.lmHead);
}

// --- Loss function ---
export function loss(params, cfg, inputIds, posIds, targetIds, vocabSize) {
  const logits = forward(params, cfg, inputIds, posIds);
  const logprobs = nn.logSoftmax(logits, -1);
  const oneHot = nn.oneHot(targetIds, vocabSize);
  return np.mean(np.sum(logprobs.mul(oneHot), -1)).neg();
}

// --- Inference (sequential, no KV cache for simplicity) ---
export async function inference(params, cfg, tokenizer, rngKey, opts = {}) {
  const temperature = opts.temperature ?? 0.5;
  const numSamples = opts.numSamples ?? 20;
  const samples = [];

  const sampleKeys = random.split(rngKey, numSamples);

  for (let s = 0; s < numSamples; s++) {
    let sampleKey = getKey(sampleKeys, s);
    let tokenIds = [tokenizer.BOS];
    const generated = [];

    for (let pos = 0; pos < cfg.blockSize; pos++) {
      const inputIds = np.array(new Int32Array(tokenIds), { dtype: np.int32 });
      const posIds = np.arange(tokenIds.length, { dtype: np.int32 });

      const logits = forward(tree.ref(params), cfg, inputIds, posIds);

      // Take logits for last position: logits[seqLen-1, :] -> [vocabSize]
      const lastLogits = logits.slice(tokenIds.length - 1);

      // Temperature scaling
      const scaled = temperature !== 1.0 ? lastLogits.div(temperature) : lastLogits;

      // Sample from categorical distribution
      const splitKeys = random.split(sampleKey, 2);
      const k1 = getKey(splitKeys, 0);
      sampleKey = getKey(splitKeys, 1);
      splitKeys.dispose();

      const chosen = random.categorical(k1, scaled);
      const chosenId = chosen.item();

      if (chosenId === tokenizer.BOS) break;
      generated.push(tokenizer.uchars[chosenId]);
      tokenIds.push(chosenId);
    }

    samples.push(generated.join(''));
  }

  sampleKeys.dispose();
  return samples;
}
