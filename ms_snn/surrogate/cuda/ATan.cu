#include "cuda_common.h"

class ATanKernelAttr : public AotKernelData {
 public:
  float alpha;
};

extern "C" int ATanBackwardInit(int *ndims, int64_t **shapes,
                                const char **dtypes, AotExtra *extra) {
  ATanKernelAttr *kernel_ptr = new ATanKernelAttr;
  kernel_ptr->alpha = extra->Attr<float>("alpha");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

__global__ static void ATanBackwardKernel(const float *x, float alpha,
                                          const float *dout, float *dx,
                                          int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float ATan_backward_alpha = M_PI / 2 * alpha * x[idx];
    dx[idx] = alpha / 2.0f /
              (1.0f + ATan_backward_alpha * ATan_backward_alpha) * dout[idx];
  }
}

extern "C" int ATanBackward(int nparam, void **params, int *ndims,
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
  auto kernel_ptr = static_cast<ATanKernelAttr *>(extra->KernelData());

  size_t size = 1;
  for (int i = 0; i < ndims[2]; i++) {
    size *= shapes[2][i];
  }
  int n = size / THREADS;

  ATanBackwardKernel<<<n + 1, THREADS, 0, custream>>>(x, kernel_ptr->alpha,
                                                      dout, dx, size);
  cudaStreamSynchronize(custream);
  return 0;
}