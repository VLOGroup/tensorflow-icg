#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

template<typename T>
using Tensor3 = tensorflow::TTypes<T,3>;

template<typename Device, typename T>
struct Fftshift2dFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor3<T>::ConstTensor &in,
                  typename Tensor3<T>::Tensor &out);
};

template<typename Device, typename T>
struct Ifftshift2dFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor3<T>::ConstTensor &in,
                  typename Tensor3<T>::Tensor &out);
};
