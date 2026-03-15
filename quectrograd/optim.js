// quectrograd Adam optimizer

export class Adam {
  constructor(params, opts = {}) {
    this.params = params;       // array of Tensors
    this.lr = opts.lr ?? 0.01;
    this.beta1 = opts.beta1 ?? 0.85;
    this.beta2 = opts.beta2 ?? 0.99;
    this.eps = opts.eps ?? 1e-8;
    this.step_t = 0;

    // Allocate m and v buffers — one flat pair per param tensor
    const backend = params[0].backend;
    this.mBuffers = params.map(p => backend.zeros(p.numel()));
    this.vBuffers = params.map(p => backend.zeros(p.numel()));
  }

  step(lr) {
    this.step_t++;
    const useLr = lr ?? this.lr;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const count = p.numel();
      p.backend.adamUpdate(
        p.handle, p.grad.handle,
        this.mBuffers[i], this.vBuffers[i],
        useLr, this.beta1, this.beta2, this.eps,
        this.step_t, count
      );
    }
  }
}
