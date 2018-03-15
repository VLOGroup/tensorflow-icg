"""Python layer for icgvn_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops
import tensorflow as tf

_icg_ops_so = loader.load_op_library(resource_loader.get_path_to_datafile("_icg_ops.so"))

fftshift2d = _icg_ops_so.fftshift2d
ifftshift2d = _icg_ops_so.ifftshift2d

activation_rbf = _icg_ops_so.activation_rbf
activation_prime_rbf = _icg_ops_so.activation_prime_rbf
activation_int_rbf = _icg_ops_so.activation_int_rbf
activation_interpolate_linear = _icg_ops_so.activation_interpolate_linear
activation_b_spline = _icg_ops_so.activation_b_spline
activation_cubic_b_spline = _icg_ops_so.activation_cubic_b_spline
activation_prime_cubic_b_spline = _icg_ops_so.activation_prime_cubic_b_spline

@ops.RegisterGradient("Fftshift2d")
def _Fftshift2dGrad(op, grad):
    in_grad = _icg_ops_so.ifftshift2d(grad)
    return [in_grad]

@ops.RegisterGradient("Ifftshift2d")
def _Iftshift2dGrad(op, grad):
    in_grad = _icg_ops_so.fftshift2d(grad)
    return [in_grad]

@ops.RegisterGradient("ActivationRBF")
def _ActivationRBFGrad(op, grad):
    rbf_prime = activation_prime_rbf(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    grad_w = _icg_ops_so.activation_rbf_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

@ops.RegisterGradient("ActivationPrimeRBF")
def _ActivationPrimeRBFGrad(op, grad):
    rbf_double_prime = _icg_ops_so.activation_double_prime_rbf(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_double_prime * grad
    grad_w = _icg_ops_so.activation_prime_rbf_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

@ops.RegisterGradient("ActivationBSpline")
def _ActivationCubicBSplineGrad(op, grad):
    rbf_prime = _icg_ops_so.activation_prime_b_spline(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    grad_w = _icg_ops_so.activation_b_spline_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

@ops.RegisterGradient("ActivationCubicBSpline")
def _ActivationCubicBSplineGrad(op, grad):
    rbf_prime = activation_prime_cubic_b_spline(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    grad_w = _icg_ops_so.activation_cubic_b_spline_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

@ops.RegisterGradient("ActivationPrimeCubicBSpline")
def _ActivationPrimeCubicBSplineGrad(op, grad):
    rbf_double_prime = _icg_ops_so.activation_double_prime_cubic_b_spline(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_double_prime * grad
    grad_w = _icg_ops_so.activation_prime_cubic_b_spline_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

@ops.RegisterGradient("ActivationInterpolateLinear")
def _ActivationInterpolateLinearGrad(op, grad):
    act_prime = _icg_ops_so.activation_prime_interpolate_linear(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = act_prime * grad
    grad_w = _icg_ops_so.activation_interpolate_linear_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

def conv2d_complex(u, k, strides=[1,1,1,1], padding='SAME', data_format='NHWC'):
    """ Complex 2d convolution with the same interface as `conv2d`.
    """
    conv_rr = tf.nn.conv2d(tf.real(u), tf.real(k),  strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ii = tf.nn.conv2d(tf.imag(u), tf.imag(k),  strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ri = tf.nn.conv2d(tf.real(u), tf.imag(k), strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ir = tf.nn.conv2d(tf.imag(u), tf.real(k), strides=strides, padding=padding,
                                     data_format=data_format)
    return tf.complex(conv_rr-conv_ii, conv_ri+conv_ir)

def conv2d_transpose_complex(u, k, output_shape, strides=[1,1,1,1], padding='SAME', data_format='NHWC'):
    """ Complex 2d transposed convolution with the same interface as `conv2d_transpose`.
    """
    convT_rr = tf.nn.conv2d_transpose(tf.real(u), tf.real(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ii = tf.nn.conv2d_transpose(tf.imag(u), tf.imag(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ri = tf.nn.conv2d_transpose(tf.real(u), tf.imag(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ir = tf.nn.conv2d_transpose(tf.imag(u), tf.real(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    return tf.complex(convT_rr+convT_ii, convT_ir-convT_ri)

def ifftc2d(inp):
    """ Centered inverse 2d Fourier transform, performed on axis (-1,-2).
    """
    shape = tf.shape(inp)
    numel = shape[-2]*shape[-1]
    scale = tf.sqrt(tf.cast(numel, tf.float32))

    out = fftshift2d(tf.ifft2d(ifftshift2d(inp)))
    out = tf.complex(tf.real(out)*scale, tf.imag(out)*scale)
    return out

def fftc2d(inp):
    """ Centered 2d Fourier transform, performed on axis (-1,-2).
    """
    shape = tf.shape(inp)
    numel = shape[-2]*shape[-1]
    scale = 1.0 / tf.sqrt(tf.cast(numel, tf.float32))

    out = fftshift2d(tf.fft2d(ifftshift2d(inp)))
    out = tf.complex(tf.real(out) * scale, tf.imag(out) * scale)
    return out