#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/contrib/icg/common/definitions.h"
#include "tensorflow/core/framework/register_types.h"

template<typename T>
__global__ void Ifftshift2dKernel(const typename tensorflow::TTypes<T,3>::ConstTensor input,
                                  typename tensorflow::TTypes<T,3>::Tensor output)
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
__global__ void Fftshift2dKernel(const typename tensorflow::TTypes<T,3>::ConstTensor input,
                                 typename tensorflow::TTypes<T,3>::Tensor output)
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

template<typename T>
void Fftshift2dKernelLauncher(const tensorflow::Tensor *in,
                              tensorflow::Tensor *out)
{
  auto kd_in = in->flat_inner_dims<T,3>();
  auto kd_out = out->flat_inner_dims<T,3>();

  dim3 dimBlock(ICGVN_BLOCK_SIZE_3D_X, ICGVN_BLOCK_SIZE_3D_Y, ICGVN_BLOCK_SIZE_3D_Z);
  dim3 dimGrid(
      divUp(kd_in.dimensions()[2], dimBlock.x),
      divUp(kd_in.dimensions()[1], dimBlock.y),
      divUp(kd_in.dimensions()[0], dimBlock.z));

  Fftshift2dKernel<T> <<<dimGrid, dimBlock>>>(kd_in, kd_out);
}

template<typename T>
void Ifftshift2dKernelLauncher(const tensorflow::Tensor *in,
               tensorflow::Tensor *out)
{
  auto kd_in = in->flat_inner_dims<T,3>();
  auto kd_out = out->flat_inner_dims<T,3>();

  dim3 dimBlock(ICGVN_BLOCK_SIZE_3D_X, ICGVN_BLOCK_SIZE_3D_Y, ICGVN_BLOCK_SIZE_3D_Z);
  dim3 dimGrid(
      divUp(kd_in.dimensions()[2], dimBlock.x),
      divUp(kd_in.dimensions()[1], dimBlock.y),
      divUp(kd_in.dimensions()[0], dimBlock.z));

  Ifftshift2dKernel<T> <<<dimGrid, dimBlock>>>(kd_in, kd_out);
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    template void Ifftshift2dKernelLauncher<T>(const tensorflow::Tensor * in, \
                                         tensorflow::Tensor * out); \
    template void Fftshift2dKernelLauncher<T>(const tensorflow::Tensor * in, \
                                         tensorflow::Tensor * out);

TF_CALL_ICG_NUMBER_TYPES(REGISTER_KERNEL_LAUNCHER);

#undef REGISTER_KERNEL_LAUNCHER

#endif
