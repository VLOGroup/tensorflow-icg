#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/icg/common/definitions.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "ffttools.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Fftshift2d")
    .Attr("T: numbertype")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(Perform 2D fftshift output = Fftshift2d(input)
)doc");

REGISTER_OP("Ifftshift2d")
    .Attr("T: numbertype")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(Perform 2D ifftshift output = Ifftshift2d(input)
)doc");

template <typename T>
struct Fftshift2dFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context,
                  const typename Tensor3<T>::ConstTensor &in,
                  typename Tensor3<T>::Tensor &out)
  {
    const int height = in.dimensions()[1];
    const int width = in.dimensions()[2];

    for(int z = 0; z < in.dimensions()[0]; z++)
    {
      for(int y = 0; y < height; y++)
      {
        for(int x = 0; x < width; x++)
        {
          int x_mid = width / 2.f;
          int y_mid = height / 2.f;

          int x_dst = (x + x_mid) % width;
          int y_dst = (y + y_mid) % height;

          out(z, y_dst, x_dst) = in(z, y, x);
        }
      }
    }
  }
};

template <typename T>
struct Ifftshift2dFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context,
                  const typename Tensor3<T>::ConstTensor &in,
                  typename Tensor3<T>::Tensor &out)
  {
    const int height = in.dimensions()[1];
    const int width = in.dimensions()[2];

    for(int z = 0; z < in.dimensions()[0]; z++)
    {
      for(int y = 0; y < height; y++)
      {
        for(int x = 0; x < width; x++)
        {
          int x_mid = (width + 1.f) / 2.f;
          int y_mid = (height + 1.f) / 2.f;

          int x_dst = (x + x_mid) % width;
          int y_dst = (y + y_mid) % height;

          out(z, y_dst, x_dst) = in(z, y, x);
        }
      }
    }
  }
};

template<typename Device, typename T>
class Fftshift2dOp : public OpKernel {
 public:
  explicit Fftshift2dOp(OpKernelConstruction* context) : OpKernel(context)
  {
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Check dimensionality
    OP_REQUIRES(context, input_tensor.dims() >= 2,
                errors::Unimplemented("Expected a >=2d Tensor, got ",
                                        input_tensor.dims(), "d."));

    // Prepare output shape
    auto output_shape = input_tensor.shape();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    // Flat inner dimensions
    auto in = input_tensor.flat_inner_dims<T,3>();
    auto out = output_tensor->flat_inner_dims<T,3>();

    // Call the kernel
    ApplyFftshift2d(context, in, out);
  }

 private:
  void ApplyFftshift2d(OpKernelContext *context,
                       const typename Tensor3<T>::ConstTensor &in,
                       typename Tensor3<T>::Tensor &out)
  {
    Fftshift2dFunctor<Device, T>()(context, in, out);
  }
};

template<typename Device, typename T>
class Ifftshift2dOp : public OpKernel {
 public:
  explicit Ifftshift2dOp(OpKernelConstruction* context) : OpKernel(context)
  {
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Check dimensionality
    OP_REQUIRES(context, input_tensor.dims() >= 2,
                errors::Unimplemented("Expected a >=2d Tensor, got ",
                                        input_tensor.dims(), "d."));

    // Prepare output shape
    auto output_shape = input_tensor.shape();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    // Flat inner dimensions
    auto in = input_tensor.flat_inner_dims<T,3>();
    auto out = output_tensor->flat_inner_dims<T,3>();

    // Call the kernel
    ApplyIfftshift2d(context, in, out);
  }

 private:
  void ApplyIfftshift2d(OpKernelContext *context,
                        const typename Tensor3<T>::ConstTensor &in,
                        typename Tensor3<T>::Tensor &out)
  {
    Ifftshift2dFunctor<Device, T>()(context, in, out);
  }
};

#define REGISTER_CPU_KERNEL(T) \
REGISTER_KERNEL_BUILDER(Name("Fftshift2d") \
                        .Device(DEVICE_CPU) \
                        .TypeConstraint<T>("T"), \
                        Fftshift2dOp<CPUDevice, T>) \
REGISTER_KERNEL_BUILDER(Name("Ifftshift2d") \
                        .Device(DEVICE_CPU) \
                        .TypeConstraint<T>("T"), \
                        Ifftshift2dOp<CPUDevice, T>)

TF_CALL_ICG_NUMBER_TYPES(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(T) \
REGISTER_KERNEL_BUILDER(Name("Fftshift2d") \
                        .Device(DEVICE_GPU) \
                        .TypeConstraint<T>("T"), \
                        Fftshift2dOp<GPUDevice, T>) \
REGISTER_KERNEL_BUILDER(Name("Ifftshift2d") \
                        .Device(DEVICE_GPU) \
                        .TypeConstraint<T>("T"), \
                        Ifftshift2dOp<GPUDevice, T>)

TF_CALL_ICG_NUMBER_TYPES(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL
#endif
