// quectrograd tape-based autograd
// Tape order IS topological order — no sorting needed.

export const tape = [];

export function backward(lossTensor) {
  // Seed gradient
  lossTensor.ensureGrad();
  lossTensor.backend.fill(lossTensor.grad.handle, 1.0, lossTensor.numel());

  // Walk tape in reverse
  for (let i = tape.length - 1; i >= 0; i--) {
    const node = tape[i];
    if (node._backwardFn) {
      node._backwardFn();
    }
  }
}

export function zeroGrad(params) {
  for (const p of params) {
    if (p.grad) {
      p.backend.fill(p.grad.handle, 0.0, p.numel());
    }
  }
}

export function clearTape() {
  tape.length = 0;
}
