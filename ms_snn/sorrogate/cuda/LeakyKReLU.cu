#include "cuda_common.h"

class LeakyKReLUKernelAttr : public AotKernelData {
 public:
  float leak;
  float k;
};

extern "C" int LeakyKReLUBackwardInit(int *ndims, int64_t **shapes,
                                      const char **dtypes, AotExtra *extra) {
  LeakyKReLUKernelAttr *kernel_ptr = new LeakyKReLUKernelAttr;
  kernel_ptr->leak = extra->Attr<float>("leak");
  kernel_ptr->k = extra->Attr<float>("k");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

__global__ static void LeakyKReLUBackwardKernel(const float *x, float leak,
                                                float k, const float *dout,
                                                float *dx, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float backward_mask = (float)(x[idx] >= 0.0f);
    dx[idx] = (k * backward_mask + leak * (1.0f - backward_mask)) * dout[idx];
  }
}

extern "C" int LeakyKReLUBackward(int nparam, void **params, int *ndims,
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
  auto kernel_ptr = static_cast<LeakyKReLUKernelAttr *>(extra->KernelData());

  size_t size = 1;
  for (int i = 0; i < ndims[2]; i++) {
    size *= shapes[2][i];
  }
  int n = size / THREADS;

  LeakyKReLUBackwardKernel<<<n + 1, THREADS, 0, custream>>>(
      x, kernel_ptr->leak, kernel_ptr->k, dout, dx, size);
  cudaStreamSynchronize(custream);
  return 0;
}