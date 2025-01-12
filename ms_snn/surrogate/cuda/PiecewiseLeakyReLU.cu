#include "cuda_common.h"

class PiecewiseLeakyReLUKernelAttr : public AotKernelData {
 public:
  float w_inv;
  float w;
  float c;
};

extern "C" int PiecewiseLeakyReLUBackwardInit(int *ndims,
                                                       int64_t **shapes,
                                                       const char **dtypes,
                                                       AotExtra *extra) {
  PiecewiseLeakyReLUKernelAttr *kernel_ptr =
      new PiecewiseLeakyReLUKernelAttr;
  float w = extra->Attr<float>("w");
  kernel_ptr->w_inv = 1.0f / w;
  kernel_ptr->w = w;
  kernel_ptr->c = extra->Attr<float>("c");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

__global__ static void PiecewiseLeakyReLUBackwardKernel(
    const float *x, float w_inv, float w, float c, const float *dout, float *dx,
    int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const float backward_mask = (float)(w >= fabsf(x[idx]));
    dx[idx] = (w_inv * backward_mask + c * (1.0f - backward_mask)) * dout[idx];
  }
}

extern "C" int PiecewiseLeakyReLUBackward(int nparam, void **params,
                                                   int *ndims, int64_t **shapes,
                                                   const char **dtypes,
                                                   void *stream,
                                                   void *extra_void) {
  constexpr int PARAM_NUM = 3;

  if (nparam != PARAM_NUM) {
    return 1;
  }
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  const float *x = static_cast<const float *>(params[0]);
  const float *dout = static_cast<const float *>(params[1]);
  float *dx = static_cast<float *>(params[2]);
  AotExtra *extra = static_cast<AotExtra *>(extra_void);
  auto kernel_ptr =
      static_cast<PiecewiseLeakyReLUKernelAttr *>(extra->KernelData());

  size_t size = 1;
  for (int i = 0; i < ndims[2]; i++) {
    size *= shapes[2][i];
  }
  int n = size / THREADS;

  PiecewiseLeakyReLUBackwardKernel<<<n + 1, THREADS, 0, custream>>>(
      x, kernel_ptr->w_inv, kernel_ptr->w, kernel_ptr->c, dout, dx, size);
  cudaStreamSynchronize(custream);
  return 0;
}