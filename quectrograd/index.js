// quectrograd public API
export { Tensor } from './tensor.js';
export { backward, zeroGrad, clearTape, tape } from './autograd.js';
export { Adam } from './optim.js';
export * as ops from './ops.js';
export { initCPU } from './backend_cpu.js';
export { initWebGPU } from './backend_webgpu.js';
