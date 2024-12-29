#include <cuda.h>
#include <cuda_runtime.h>

#include "custom_aot_extra.h"

constexpr int THREADS = 1024;

class atan_kernel_attr : public AotKernelData {
 public:
  float alpha;
};

__global__ void AtanForwardKernel(const float *x, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = (x[idx] >= 0.0) ? 1.0f : 0.0f;
  }
}

extern "C" int AtanForward(int nparam, void **params, int *ndims,
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

  AtanForwardKernel<<<n + 1, THREADS, 0, custream>>>(x, out, size);
  cudaStreamSynchronize(custream);
  return 0;
}

extern "C" int AtanBackwardInit(int *ndims, int64_t **shapes,
                                const char **dtypes, AotExtra *extra) {
  atan_kernel_attr *kernel_ptr = new atan_kernel_attr;
  kernel_ptr->alpha = extra->Attr<float>("alpha");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

__global__ void AtanBackwardKernel(const float *x, float alpha,
                                   const float *dout, float *dx, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float atan_backward_alpha = M_PI / 2 * alpha * x[idx];
    dx[idx] = alpha / 2.0f /
              (1.0f + atan_backward_alpha * atan_backward_alpha) * dout[idx];
  }
}

extern "C" int AtanBackward(int nparam, void **params, int *ndims,
                            int64_t **shapes, const char **dtypes, void *stream,
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
  auto kernel_ptr = static_cast<atan_kernel_attr *>(extra->KernelData());

  size_t size = 1;
  for (int i = 0; i < ndims[2]; i++) {
    size *= shapes[2][i];
  }
  int n = size / THREADS;

  AtanBackwardKernel<<<n + 1, THREADS, 0, custream>>>(x, kernel_ptr->alpha,
                                                      dout, dx, size);
  cudaStreamSynchronize(custream);
  return 0;
}