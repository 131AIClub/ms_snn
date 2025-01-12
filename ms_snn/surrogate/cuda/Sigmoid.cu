#include "cuda_common.h"

class SigmoidKernelAttr : public AotKernelData {
 public:
  float alpha;
};

extern "C" int SigmoidBackwardInit(int *ndims, int64_t **shapes,
                                   const char **dtypes, AotExtra *extra) {
  SigmoidKernelAttr *kernel_ptr = new SigmoidKernelAttr;
  kernel_ptr->alpha = extra->Attr<float>("alpha");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

__global__ static void SigmoidBackwardKernel(const float *x, float alpha,
                                             const float *dout, float *dx,
                                             int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float sigmoid_ax = 1.0f / (1.0f + expf(-alpha) * x[idx]);
    dx[idx] = ((1.0f - sigmoid_ax) * sigmoid_ax * alpha) * dout[idx];
  }
}

extern "C" int SigmoidBackward(int nparam, void **params, int *ndims,
                               int64_t **shapes, const char **dtypes,
                               void *stream, void *extra_void) {
  constexpr int PARAM_NUM = 3;

  if (nparam != PARAM_NUM) {
    return 1;
  }
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  const float *x = static_cast<const float *>(params[0]);
  const float *dout = static_cast<const float *>(params[1]);
  float *dx = static_cast<float *>(params[2]);
  AotExtra *extra = static_cast<AotExtra *>(extra_void);
  auto kernel_ptr = static_cast<SigmoidKernelAttr *>(extra->KernelData());

  size_t size = 1;
  for (int i = 0; i < ndims[2]; i++) {
    size *= shapes[2][i];
  }
  int n = size / THREADS;

  SigmoidBackwardKernel<<<n + 1, THREADS, 0, custream>>>(x, kernel_ptr->alpha,
                                                         dout, dx, size);
  cudaStreamSynchronize(custream);
  return 0;
}