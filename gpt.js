// quectoGPT model definition — pure functions, no side effects
// Uses jax-js for tensor operations + autodiff

import { numpy as np, nn, random, tree, blockUntilReady } from '@jax-js/jax';

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

// --- Parameter initialization ---
// Weights stored as [inDim, outDim] — no transpose needed in forward pass
export async function initParams(vocabSize, cfg, rngKey) {
  const nEmbd = cfg.nEmbd;
  const s = Math.sqrt(3 / nEmbd);
  const numKeys = 3 + cfg.nLayer * 6;

  const keys = random.split(rngKey, numKeys);
  let ki = 0;
  const nextKey = () => { ki++; return ki < numKeys ? keys.ref.slice(ki - 1) : keys.slice(ki - 1); };

  const params = {
    wte: random.normal(nextKey(), [vocabSize, nEmbd]).mul(cfg.initStd),
    wpe: random.normal(nextKey(), [cfg.blockSize, nEmbd]).mul(cfg.initStd),
    lmHead: random.normal(nextKey(), [nEmbd, vocabSize]).mul(0.001),
    layers: [],
  };

  for (let i = 0; i < cfg.nLayer; i++) {
    params.layers.push({
      wq: random.uniform(nextKey(), [nEmbd, nEmbd], { minval: -s, maxval: s }),
      wk: random.uniform(nextKey(), [nEmbd, nEmbd], { minval: -s, maxval: s }),
      wv: random.uniform(nextKey(), [nEmbd, nEmbd], { minval: -s, maxval: s }),
      wo: np.zeros([nEmbd, nEmbd]),
      mlpFc1: random.uniform(nextKey(), [nEmbd, 4 * nEmbd], { minval: -0.4 * s, maxval: 0.4 * s }),
      mlpFc2: np.zeros([4 * nEmbd, nEmbd]),
    });
  }

  await blockUntilReady(params);
  return params;
}

// --- RMSNorm (no learnable params) ---
function rmsnorm(x) {
  const ms = np.mean(np.square(x.ref), -1, { keepdims: true });
  return x.div(np.sqrt(ms.add(1e-5)));
}

// --- Forward pass ---
// tokenOH/posOH: pre-computed oneHot matrices
//   Unbatched: [seqLen, vocabSize] and [seqLen, blockSize]
//   Batched:   [B, seqLen, vocabSize] and [B, seqLen, blockSize]
// seqLen: sequence length (JS number, needed for reshape)
// batched: whether input has a batch dimension (default false)
export function forward(params, cfg, tokenOH, posOH, seqLen, batched = false) {
  const headDim = cfg.nEmbd / cfg.nHead;
  const reshapeQKV = batched
    ? (t) => t.reshape([-1, seqLen, cfg.nHead, headDim])
    : (t) => t.reshape([seqLen, cfg.nHead, headDim]);
  const reshapeAttn = batched
    ? (t) => t.reshape([-1, seqLen, cfg.nEmbd])
    : (t) => t.reshape([seqLen, cfg.nEmbd]);

  // Token + position embeddings via oneHot @ weight
  let x = np.dot(tokenOH, params.wte);     // [..., seqLen, nEmbd]
  const posEmb = np.dot(posOH, params.wpe); // [..., seqLen, nEmbd]
  x = rmsnorm(x.add(posEmb));

  for (let li = 0; li < cfg.nLayer; li++) {
    const layer = params.layers[li];
    const xResidual = x.ref;
    x = rmsnorm(x);

    // QKV projections: [..., seqLen, nEmbd] @ [nEmbd, nEmbd] -> [..., seqLen, nEmbd]
    const q = np.dot(x.ref, layer.wq);
    const k = np.dot(x.ref, layer.wk);
    const v = np.dot(x, layer.wv);

    // Reshape to [..., seqLen, nHead, headDim] for dotProductAttention
    const qH = reshapeQKV(q);
    const kH = reshapeQKV(k);
    const vH = reshapeQKV(v);

    // Scaled dot-product attention with causal mask
    const attnOut = nn.dotProductAttention(qH, kH, vH, { isCausal: true });

    // Reshape back to [..., seqLen, nEmbd] and project
    const attnFlat = reshapeAttn(attnOut);
    x = np.dot(attnFlat, layer.wo).add(xResidual);

    // MLP block
    const mlpResidual = x.ref;
    x = rmsnorm(x);
    x = nn.relu(np.dot(x, layer.mlpFc1));     // [..., seqLen, 4*nEmbd]
    x = np.dot(x, layer.mlpFc2).add(mlpResidual); // [..., seqLen, nEmbd]
  }

  // Output logits: [..., seqLen, vocabSize]
  return np.dot(x, params.lmHead);
}

// --- Loss function ---
// tokenOH, posOH, targetOH: pre-computed oneHot matrices (outside valueAndGrad)
// mask: optional [..., seqLen] binary mask (1 = real token, 0 = padding)
export function loss(params, cfg, tokenOH, posOH, targetOH, seqLen, mask, batched = false) {
  const logits = forward(params, cfg, tokenOH, posOH, seqLen, batched);
  const logprobs = nn.logSoftmax(logits, -1);
  const perToken = np.sum(logprobs.mul(targetOH), -1).neg(); // [..., seqLen]
  if (mask) {
    // Masked mean: only count real (non-padded) tokens
    return np.sum(perToken.mul(mask)).div(np.sum(mask));
  }
  return np.mean(perToken);
}

// --- Inference (sequential, no KV cache for simplicity) ---
export async function inference(params, cfg, tokenizer, rngKey, opts = {}) {
  const temperature = opts.temperature ?? 0.5;
  const numSamples = opts.numSamples ?? 20;
  const samples = [];

  const sampleKeys = random.split(rngKey, numSamples);

  for (let s = 0; s < numSamples; s++) {
    let sampleKey = sampleKeys.ref.slice(s);
    let tokenIds = [tokenizer.BOS];
    const generated = [];

    for (let pos = 0; pos < cfg.blockSize; pos++) {
      const seqLen = tokenIds.length;
      const inputIds = np.array(new Int32Array(tokenIds), { dtype: np.int32 });
      const posIds = np.arange(seqLen).astype(np.int32);
      const tokenOH = nn.oneHot(inputIds, tokenizer.vocabSize);
      const posOH = nn.oneHot(posIds, cfg.blockSize);

      const logits = forward(tree.ref(params), cfg, tokenOH, posOH, seqLen, false);

      // Take logits for last position: logits[seqLen-1, :] -> [vocabSize]
      const lastLogits = logits.slice(seqLen - 1);

      // Temperature scaling
      const scaled = temperature !== 1.0 ? lastLogits.div(temperature) : lastLogits;

      // Sample from categorical distribution
      const splitKeys = random.split(sampleKey, 2);
      const k1 = splitKeys.ref.slice(0);
      sampleKey = splitKeys.slice(1);

      const chosen = random.categorical(k1, scaled);
      const chosenId = chosen.item();

      if (chosenId === tokenizer.BOS || chosenId === tokenizer.EOS) break;
      generated.push(chosenId);
      tokenIds.push(chosenId);
    }

    samples.push(tokenizer.decode(generated));
  }

  sampleKeys.dispose();
  return samples;
}
