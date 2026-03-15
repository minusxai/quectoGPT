// tokenize.js - Train BPE tokenizer and pre-tokenize dataset into binary files
// Usage: deno run --allow-read --allow-write --allow-env tokenize.js --dataset=names [--vocab=1024]

import { trainBPE, buildBPETokenizer } from './bpe.js';

async function main() {
  const isDeno = typeof Deno !== 'undefined';
  const args = isDeno ? Deno.args : process.argv.slice(2);
  const getArg = (name, def) => {
    const a = args.find(x => x.startsWith(`--${name}=`));
    return a ? a.split('=')[1] : def;
  };

  const dataset = getArg('dataset', 'names');
  const targetVocab = parseInt(getArg('vocab', '1024'));
  const numMerges = targetVocab - 256 - 2; // 256 bytes + numMerges tokens + BOS + EOS
  const valFrac = parseFloat(getArg('val', '0.1'));

  const dataPath = `data/${dataset}.txt`;

  console.log(`=== BPE Tokenizer Training ===\n`);
  console.log(`dataset:      ${dataset}`);
  console.log(`data path:    ${dataPath}`);
  console.log(`target vocab: ${targetVocab}`);
  console.log(`num merges:   ${numMerges}`);
  console.log(`val fraction: ${valFrac}\n`);

  // Read data
  const text = isDeno
    ? Deno.readTextFileSync(dataPath)
    : (await import('fs')).readFileSync(dataPath, 'utf-8');

  const textBytes = new TextEncoder().encode(text);
  console.log(`raw text:     ${text.length} chars, ${textBytes.length} bytes\n`);

  // Train BPE on full text
  console.log(`training BPE...`);
  const t0 = performance.now();
  const merges = trainBPE(textBytes, numMerges);
  const trainTime = (performance.now() - t0) / 1000;
  console.log(`\ntrained ${merges.length} merges in ${trainTime.toFixed(2)}s\n`);

  // Build tokenizer
  const tok = buildBPETokenizer(merges);

  // Split into train/val by lines
  const lines = text.split('\n').filter(l => l.length > 0);
  const valCount = Math.max(1, Math.floor(lines.length * valFrac));
  const trainLines = lines.slice(0, lines.length - valCount);
  const valLines = lines.slice(lines.length - valCount);

  // Encode each line with BOS/EOS, then concatenate into flat token stream
  function encodeLines(lineArr) {
    const allTokens = [];
    for (const line of lineArr) {
      const encoded = tok.encode(line);
      for (const t of encoded) allTokens.push(t);
    }
    return allTokens;
  }

  const trainTokens = encodeLines(trainLines);
  const valTokens = encodeLines(valLines);

  // --- Metrics ---
  const trainChars = trainLines.reduce((s, l) => s + l.length, 0);
  const valChars = valLines.reduce((s, l) => s + l.length, 0);
  const trainBytes = new TextEncoder().encode(trainLines.join('\n')).length;
  const valBytes = new TextEncoder().encode(valLines.join('\n')).length;
  const trainRatio = trainBytes / trainTokens.length;
  const valRatio = valBytes / valTokens.length;

  console.log(`=== Tokenizer Metrics ===\n`);
  console.log(`vocab size:         ${tok.vocabSize}`);
  console.log(`BOS token:          ${tok.BOS}`);
  console.log(`EOS token:          ${tok.EOS}`);
  console.log(`actual merges:      ${merges.length}`);
  console.log(`\ntrain split:        ${trainLines.length} lines, ${trainChars} chars, ${trainBytes} bytes`);
  console.log(`train tokens:       ${trainTokens.length} (${trainRatio.toFixed(2)} bytes/token, ${(trainChars / trainTokens.length).toFixed(2)} chars/token)`);
  console.log(`\nval split:          ${valLines.length} lines, ${valChars} chars, ${valBytes} bytes`);
  console.log(`val tokens:         ${valTokens.length} (${valRatio.toFixed(2)} bytes/token, ${(valChars / valTokens.length).toFixed(2)} chars/token)`);

  // Roundtrip check
  const sampleLine = lines[0];
  const encoded = tok.encode(sampleLine);
  const decoded = tok.decode(encoded);
  const roundtripOk = decoded === sampleLine;
  console.log(`\nroundtrip check:    "${sampleLine}" -> [${encoded.length} tokens] -> "${decoded}" ${roundtripOk ? '  OK' : '  FAIL'}`);

  // Top merges
  console.log(`\n=== Top 20 Merges ===\n`);
  for (let i = 0; i < Math.min(20, merges.length); i++) {
    const [a, b] = merges[i];
    const safeStr = (id) => {
      try {
        const bytes = tok.vocab[id];
        const s = new TextDecoder().decode(bytes);
        // Only show printable ASCII
        return /^[\x20-\x7e]+$/.test(s) ? `"${s}"` : `<${id}>`;
      } catch { return `<${id}>`; }
    };
    const mergedStr = safeStr(256 + i);
    console.log(`  ${String(i + 1).padStart(2)}. ${safeStr(a).padEnd(8)} + ${safeStr(b).padEnd(8)} -> ${mergedStr.padEnd(10)} (id=${256 + i})`);
  }

  // --- Save files ---
  const tokJson = {
    merges,
    vocab_size: tok.vocabSize,
    bos: tok.BOS,
    eos: tok.EOS,
    dataset,
    metrics: {
      train_tokens: trainTokens.length,
      val_tokens: valTokens.length,
      train_lines: trainLines.length,
      val_lines: valLines.length,
      bytes_per_token: trainRatio,
      chars_per_token: trainChars / trainTokens.length,
    },
  };

  const tokPath = `data/${dataset}.tok.json`;
  const trainBinPath = `data/${dataset}.train.bin`;
  const valBinPath = `data/${dataset}.val.bin`;

  const trainBin = new Uint16Array(trainTokens);
  const valBin = new Uint16Array(valTokens);

  if (isDeno) {
    Deno.writeTextFileSync(tokPath, JSON.stringify(tokJson));
    Deno.writeFileSync(trainBinPath, new Uint8Array(trainBin.buffer));
    Deno.writeFileSync(valBinPath, new Uint8Array(valBin.buffer));
  } else {
    const fs = await import('fs');
    fs.writeFileSync(tokPath, JSON.stringify(tokJson));
    fs.writeFileSync(trainBinPath, Buffer.from(trainBin.buffer));
    fs.writeFileSync(valBinPath, Buffer.from(valBin.buffer));
  }

  console.log(`\n=== Files Written ===\n`);
  console.log(`  ${tokPath}       (${JSON.stringify(tokJson).length} bytes)`);
  console.log(`  ${trainBinPath}   (${trainBin.byteLength} bytes, ${trainTokens.length} tokens)`);
  console.log(`  ${valBinPath}     (${valBin.byteLength} bytes, ${valTokens.length} tokens)`);
}

main().catch(console.error);
