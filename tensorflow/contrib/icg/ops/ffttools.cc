#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/icg/common/definitions.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("Fftshift2d")
    .Attr("T: numbertype")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(Perform 2D fftshift
  output = Fftshift2d(input)
)doc");

REGISTER_OP("Ifftshift2d")
    .Attr("T: numbertype")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(Perform 2D ifftshift output = Ifftshift2d(input)
)doc");

template<typename T>
void Fftshift2dKernelLauncher(const Tensor * in, Tensor * out);
template<typename T>
void Ifftshift2dKernelLauncher(const Tensor * in, Tensor * out);

template<typename T>
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

    // Call the cuda kernel launcher
    Fftshift2dKernelLauncher<T>(&input_tensor, output_tensor);
  }
};

template<typename T>
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

    // Call the cuda kernel launcher
    Ifftshift2dKernelLauncher<T>(&input_tensor, output_tensor);
  }
};

#define REGISTER_GPU_KERNEL(T) \
REGISTER_KERNEL_BUILDER(Name("Fftshift2d") \
                        .Device(DEVICE_GPU) \
                        .TypeConstraint<T>("T"), \
                        Fftshift2dOp<T>) \
REGISTER_KERNEL_BUILDER(Name("Ifftshift2d") \
                        .Device(DEVICE_GPU) \
                        .TypeConstraint<T>("T"), \
                        Ifftshift2dOp<T>)

TF_CALL_ICG_NUMBER_TYPES(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL
