#include "cuda_common.h"

__global__ static void HeavisideKernel(const float *x, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = (x[idx] >= 0.0) ? 1.0f : 0.0f;
  }
}

extern "C" int Heaviside(int nparam, void **params, int *ndims,
                         int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
  constexpr int PARAM_NUM = 2;

  if (nparam != PARAM_NUM) {
    return 1;
  }
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  const float *x = static_cast<const float *>(params[0]);
  float *out = static_cast<float *>(params[1]);

  size_t size = 1;
  for (int i = 0; i < ndims[1]; i++) {
    size *= shapes[1][i];
  }
  int n = size / THREADS;

  HeavisideKernel<<<n + 1, THREADS, 0, custream>>>(x, out, size);
  cudaStreamSynchronize(custream);
  return 0;
}