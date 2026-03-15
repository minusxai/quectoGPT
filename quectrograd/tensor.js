// quectrograd Tensor class — backend-agnostic

export class Tensor {
  constructor(handle, shape, backend, opts = {}) {
    this.handle = handle;           // Float32Array (CPU) or GPUBuffer (WebGPU)
    this.shape = shape;             // e.g. [16, 16]
    this.backend = backend;         // reference to active backend
    this.grad = null;               // Tensor | null
    this.requiresGrad = opts.requiresGrad ?? false;
    this._op = null;                // string: 'matmul', 'add', etc.
    this._parents = [];             // Tensor[] inputs
    this._backwardFn = null;        // function to compute grads for parents
    this._saved = {};               // saved tensors/values for backward
  }

  static from(data, shape, backend, opts = {}) {
    const handle = backend.fromArray(data);
    return new Tensor(handle, shape, backend, opts);
  }

  static zeros(shape, backend, opts = {}) {
    const size = shape.reduce((a, b) => a * b, 1);
    const handle = backend.zeros(size);
    return new Tensor(handle, shape, backend, opts);
  }

  static randn(shape, std, backend, opts = {}) {
    const size = shape.reduce((a, b) => a * b, 1);
    const handle = backend.randn(size, std);
    return new Tensor(handle, shape, backend, opts);
  }

  numel() {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  // Returns Float32Array. Async for GPU (readback), sync-compatible for CPU.
  // Always use `await tensor.toArray()` — works for both backends.
  toArray() {
    return this.backend.toArray(this.handle, this.numel());
  }

  // Ensure grad tensor is allocated and zeroed
  ensureGrad() {
    if (!this.grad) {
      this.grad = Tensor.zeros(this.shape, this.backend);
    }
  }
}
