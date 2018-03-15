#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/contrib/icg/common/definitions.h"
#include "tensorflow/core/framework/register_types.h"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>

namespace tficg {

unsigned int nextPowerof2(unsigned int v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

template<typename T, int NDIMS>
void fill(const GPUDevice &d,
          typename tensorflow::TTypes<T,NDIMS>::Tensor &x, T value)
{
  thrust::fill(thrust::cuda::par.on(d.stream()),
               thrust::device_ptr<T>(x.data()),
               thrust::device_ptr<T>(x.data() + x.size()),
               value);
}

#define REGISTER_GPU_FILL(T) \
    template void fill<T, 1>(const GPUDevice &d, tensorflow::TTypes<T, 1>::Tensor &x, T value); \
    template void fill<T, 2>(const GPUDevice &d, tensorflow::TTypes<T, 2>::Tensor &x, T value); \
    template void fill<T, 3>(const GPUDevice &d, tensorflow::TTypes<T, 3>::Tensor &x, T value); \
    template void fill<T, 4>(const GPUDevice &d, tensorflow::TTypes<T, 4>::Tensor &x, T value); \
    template void fill<T, 5>(const GPUDevice &d, tensorflow::TTypes<T, 5>::Tensor &x, T value); \
    template void fill<T, 6>(const GPUDevice &d, tensorflow::TTypes<T, 6>::Tensor &x, T value);
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FILL);
#undef REGISTER_GPU_FILL

}

#endif
