// bpe.js - Byte-Pair Encoding tokenizer (training + encode/decode)
// Based on Karpathy's minbpe BasicTokenizer — byte-level BPE, no regex splitting.

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// --- BPE Training ---
// Learns `numMerges` merge rules from raw bytes.
// Returns array of [tokenA, tokenB] pairs in merge order.
export function trainBPE(textBytes, numMerges, verbose = true) {
  let ids = Array.from(textBytes);
  const originalLen = ids.length;
  const merges = [];

  for (let m = 0; m < numMerges; m++) {
    // Count adjacent pairs
    const counts = new Map();
    for (let i = 0; i < ids.length - 1; i++) {
      const key = ids[i] * 65536 + ids[i + 1];
      counts.set(key, (counts.get(key) || 0) + 1);
    }

    if (counts.size === 0) break;

    // Find most frequent pair
    let bestKey = -1, bestCount = 0;
    for (const [key, count] of counts) {
      if (count > bestCount) { bestCount = count; bestKey = key; }
    }

    if (bestCount < 2) break; // no useful merges left

    const a = Math.floor(bestKey / 65536);
    const b = bestKey % 65536;
    const newId = 256 + m;

    // Apply merge in-place
    ids = applyMerge(ids, a, b, newId);
    merges.push([a, b]);

    if (verbose && ((m + 1) % 100 === 0 || m === numMerges - 1)) {
      const ratio = originalLen / ids.length;
      console.log(`  merge ${String(m + 1).padStart(4)}/${numMerges} | (${a}, ${b}) -> ${newId} | count ${bestCount} | tokens ${ids.length} | ratio ${ratio.toFixed(2)}x`);
    }
  }

  return merges;
}

// --- Build tokenizer from merges ---
export function buildBPETokenizer(merges) {
  const BOS = 256 + merges.length;
  const EOS = BOS + 1;
  const vocabSize = EOS + 1;

  // Build vocab: token id -> byte sequence
  const vocab = new Array(vocabSize);
  for (let i = 0; i < 256; i++) vocab[i] = new Uint8Array([i]);
  for (let i = 0; i < merges.length; i++) {
    const [a, b] = merges[i];
    const aBytes = vocab[a];
    const bBytes = vocab[b];
    const merged = new Uint8Array(aBytes.length + bBytes.length);
    merged.set(aBytes);
    merged.set(bBytes, aBytes.length);
    vocab[256 + i] = merged;
  }

  function encode(text) {
    let ids = Array.from(encoder.encode(text));
    for (let m = 0; m < merges.length; m++) {
      const [a, b] = merges[m];
      ids = applyMerge(ids, a, b, 256 + m);
    }
    return [BOS, ...ids, EOS];
  }

  function decode(ids) {
    const chunks = [];
    for (const id of ids) {
      if (id === BOS || id === EOS || id < 0 || id >= vocabSize) continue;
      chunks.push(vocab[id]);
    }
    const totalLen = chunks.reduce((s, c) => s + c.length, 0);
    const bytes = new Uint8Array(totalLen);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    return decoder.decode(bytes);
  }

  return { encode, decode, BOS, EOS, vocabSize, merges, vocab };
}

function applyMerge(ids, a, b, newId) {
  const out = [];
  let i = 0;
  while (i < ids.length) {
    if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
      out.push(newId);
      i += 2;
    } else {
      out.push(ids[i]);
      i++;
    }
  }
  return out;
}
