#pragma once

#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define ICGVN_BLOCK_SIZE_3D_X 16
#define ICGVN_BLOCK_SIZE_3D_Y 8
#define ICGVN_BLOCK_SIZE_3D_Z 4

#define ICGVN_BLOCK_SIZE_2D_X 32
#define ICGVN_BLOCK_SIZE_2D_Y 32

#define ICGVN_RBF_BLOCK_SIZE_X 8
#define ICGVN_RBF_BLOCK_SIZE_Y 8

#if GOOGLE_CUDA
  #define EIGEN_USE_GPU
  #define DEVICE_PREFIX __device__
#else
  #define DEVICE_PREFIX
#endif

#define TF_CALL_ICG_REAL_NUMBER_TYPES(m)                    \
   TF_CALL_float(m) TF_CALL_double(m)
#define TF_CALL_ICG_COMPLEX_NUMBER_TYPES(m)                 \
   TF_CALL_complex64(m) TF_CALL_complex128(m)
#define TF_CALL_ICG_NUMBER_TYPES(m)                         \
   TF_CALL_ICG_REAL_NUMBER_TYPES(m) TF_CALL_ICG_COMPLEX_NUMBER_TYPES(m)

inline int divUp(int length, int block_size)
{
  return (length + block_size - 1) / block_size;
}

using GPUDevice = Eigen::GpuDevice;

namespace tficg {

unsigned int nextPowerof2(unsigned int v);

template<typename T, int NDIMS>
void fill(const GPUDevice &d,
          typename tensorflow::TTypes<T,NDIMS>::Tensor &x, T value);

}
