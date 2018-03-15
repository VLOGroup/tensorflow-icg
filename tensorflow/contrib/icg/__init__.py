"""##Ops developed at ICG.

### API

This module provides functions for building functions developed at ICG.

## Icg `Ops`

@@conv2d_complex
@@conv2d_complex_transpose
@@iffc2d
@@fftc2d
@@fftshift2d
@@ifftshift2d
@@activation_rbf
@@activation_prime_rbf
@@activation_b_spline
@@activation_cubic_b_spline
@@activation_prime_cubic_b_spline
@@activation_interpolate_linear

## Classes
@@Variational Network
@@VnBasicCell
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Operators
from tensorflow.contrib.icg.python.ops.icg_ops import conv2d_complex, conv2d_transpose_complex
from tensorflow.contrib.icg.python.ops.icg_ops import ifftc2d, fftc2d
from tensorflow.contrib.icg.python.ops.icg_ops import fftshift2d, ifftshift2d
from tensorflow.contrib.icg.python.ops.icg_ops import activation_rbf, activation_prime_rbf, activation_int_rbf
from tensorflow.contrib.icg.python.ops.icg_ops import activation_b_spline
from tensorflow.contrib.icg.python.ops.icg_ops import activation_cubic_b_spline, activation_prime_cubic_b_spline
from tensorflow.contrib.icg.python.ops.icg_ops import activation_interpolate_linear
from tensorflow.contrib.icg.python import optimizer
from tensorflow.contrib.icg.python import utils
from tensorflow.contrib.icg.python.variationalnetwork import VariationalNetwork, VnBasicCell