#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/contrib/icg/common/definitions.h"
#include "tensorflow/core/framework/register_types.h"

#include "tensorflow/contrib/icg/ops/ffttools.h"

template<typename T>
__global__ void Ifftshift2dKernel(const typename Tensor3<T>::ConstTensor input,
                                  typename Tensor3<T>::Tensor output)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  const int height = input.dimensions()[1];
  const int width = input.dimensions()[2];

  int x_mid = (width + 1.f) / 2.f;
  int y_mid = (height + 1.f) / 2.f;

  if (x < width && y < height && z < input.dimensions()[0])
  {
    int x_dst = (x + x_mid) % width;
    int y_dst = (y + y_mid) % height;

    output(z, y_dst, x_dst) = input(z, y, x);
  }
}

template<typename T>
__global__ void Fftshift2dKernel(const typename Tensor3<T>::ConstTensor input,
                                 typename Tensor3<T>::Tensor output)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  const int height = input.dimensions()[1];
  const int width = input.dimensions()[2];

  int x_mid = width / 2.f;
  int y_mid = height / 2.f;

  if (x < width && y < height && z < input.dimensions()[0])
  {
    int x_dst = (x + x_mid) % width;
    int y_dst = (y + y_mid) % height;

    output(z, y_dst, x_dst) = input(z, y, x);
  }
}

template <typename T>
struct Fftshift2dFunctor<GPUDevice, T> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor3<T>::ConstTensor &in,
                  typename Tensor3<T>::Tensor &out)
  {
    dim3 dimBlock(ICGVN_BLOCK_SIZE_3D_X, ICGVN_BLOCK_SIZE_3D_Y, ICGVN_BLOCK_SIZE_3D_Z);
    dim3 dimGrid(
        divUp(in.dimensions()[2], dimBlock.x),
        divUp(in.dimensions()[1], dimBlock.y),
        divUp(in.dimensions()[0], dimBlock.z));

    Fftshift2dKernel<T> <<<dimGrid, dimBlock>>>(in, out);
  }
};

template <typename T>
struct Ifftshift2dFunctor<GPUDevice, T> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor3<T>::ConstTensor &in,
                  typename Tensor3<T>::Tensor &out)
{
  dim3 dimBlock(ICGVN_BLOCK_SIZE_3D_X, ICGVN_BLOCK_SIZE_3D_Y, ICGVN_BLOCK_SIZE_3D_Z);
  dim3 dimGrid(
      divUp(in.dimensions()[2], dimBlock.x),
      divUp(in.dimensions()[1], dimBlock.y),
      divUp(in.dimensions()[0], dimBlock.z));

  Ifftshift2dKernel<T> <<<dimGrid, dimBlock>>>(in, out);
}
};

#define REGISTER_GPU_FUNCTOR(T) \
template struct  Fftshift2dFunctor<GPUDevice, T>; \
template struct Ifftshift2dFunctor<GPUDevice, T>;
TF_CALL_ICG_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR

#endif
