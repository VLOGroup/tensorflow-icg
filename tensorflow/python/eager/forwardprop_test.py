# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import weakref

from absl.testing import parameterized
import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import forwardprop
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import test
from tensorflow.python.util import nest


_X11_35_DERIVATIVES = [
    1.1 ** 3.5,
    3.5 * 1.1 ** 2.5,
    3.5 * 2.5 * 1.1 ** 1.5,
    3.5 * 2.5 * 1.5 * 1.1 ** 0.5]


# TODO(allenl): Move this somewhere useful once forward gradients are stable.
def _jvp(f, primals, tangents):
  """Compute the jacobian of `f` at `primals` multiplied by `tangents`."""
  with forwardprop.ForwardGradientAccumulator() as acc:
    acc.watch(primals, tangents)
    primals_out = f(*primals)
  return primals_out, acc.jvp(primals_out)


def _jacfwd(f, primals):
  """Compute the jacobian of `f` at `primals` using forward-mode autodiff."""
  jac_flat = []
  flat_primals = nest.flatten(primals)
  tangent_mask = [array_ops.zeros_like(primal) for primal in flat_primals]
  for primal_index, primal in enumerate(flat_primals):
    primal_vector = array_ops.reshape(primal, [-1])
    primal_vector_length = array_ops.size(primal_vector)
    jac_columns = []
    for element_index in math_ops.range(primal_vector_length):
      mask = array_ops.one_hot(element_index, primal_vector_length)
      tangent_mask[primal_index] = array_ops.reshape(mask,
                                                     array_ops.shape(primal))
      jac_columns.append(
          nest.map_structure(
              functools.partial(array_ops.reshape, shape=[-1]),
              _jvp(f, primals, tangent_mask)[1]))
    jac_flat.append(array_ops.stack(jac_columns, axis=1))
    tangent_mask[primal_index] = array_ops.zeros_like(primal)
  return nest.pack_sequence_as(primals, jac_flat)


def _grad(f, argnums=0):
  """Return a function which computes the gradient of `f`."""

  def _f(*params):
    with backprop.GradientTape() as tape:
      tape.watch(params)
      primals_out = f(*params)
    return tape.gradient(
        primals_out,
        params[argnums],
        unconnected_gradients=UnconnectedGradients.ZERO)

  return _f


def _hvp(f, primals, tangents):
  """Compute a forward-over-back Hessian-vector product."""
  return _jvp(_grad(f), primals, tangents)[1]


def _test_gradients(testcase,
                    f,
                    primals,
                    order,
                    delta=1e-3,
                    rtol=1e-2,
                    atol=1e-6):
  """Tests forward/backward jacobians of `f`'s [0, `order`)-order gradients."""
  if order < 1:
    raise ValueError(
        "`order` should be a positive integer, got '{}'.".format(order))
  if order > 1:
    _test_gradients(
        testcase=testcase,
        f=_grad(f),
        primals=primals,
        order=order - 1,
        delta=delta,
        rtol=rtol,
        atol=atol)
  sym_jac_back, num_jac = gradient_checker_v2.compute_gradient(
      f, primals, delta=delta)
  testcase.assertAllClose(num_jac, sym_jac_back, rtol=rtol, atol=atol)
  # TODO(b/134972215): compute_gradient should use the definition of a Jacobian
  # matrix on Wikipedia, then this transpose can go away.
  sym_jac_fwd = nest.map_structure(array_ops.transpose, _jacfwd(f, primals))
  testcase.assertAllClose(num_jac, sym_jac_fwd, rtol=rtol, atol=atol)
  # And the symbolic computations should be much closer.
  testcase.assertAllClose(sym_jac_back, sym_jac_fwd)


class ForwardpropTest(test.TestCase, parameterized.TestCase):

  def testForwardGradientFunction(self):
    add_outputs = (constant_op.constant(4.),)
    vp, = forwardprop._forward_gradient(
        op_name="Add",
        attr_tuple=(),
        inputs=(constant_op.constant(1.), constant_op.constant(3.)),
        outputs=add_outputs,
        tangents=(
            constant_op.constant(1.),
            constant_op.constant(5.),
        ))
    self.assertAllClose(1. + 5., self.evaluate(vp))

    mul_outputs = (constant_op.constant([20.]),)
    vp, = forwardprop._forward_gradient(
        op_name="Mul",
        attr_tuple=(),
        inputs=(constant_op.constant([4.]), constant_op.constant([5.])),
        outputs=mul_outputs,
        tangents=(
            constant_op.constant([2.]),
            constant_op.constant([3.]),
        ))
    self.assertAllClose([2. * 5. + 3. * 4.], self.evaluate(vp))

  def testForwardGradientFunctionUsedByAccumulatorForOps(self):
    previous_fn = forwardprop._forward_gradient
    try:
      with forwardprop.ForwardGradientAccumulator() as acc:
        x = constant_op.constant(1.)
        acc.watch(x, 2.)
        y = x + x
        pywrap_tensorflow.TFE_Py_RegisterForwardGradientFunction(
            lambda *args, **kwargs: [constant_op.constant(-15.)])
        z = x + x
      self.assertAllClose(4., acc.jvp(y))
      self.assertAllClose(-15., acc.jvp(z))
    finally:
      pywrap_tensorflow.TFE_Py_RegisterForwardGradientFunction(previous_fn)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testFunctionCacheLimited(self):
    # Every time this test is executed, it will create a slightly larger Tensor
    # and push it through Add's gradient. Since we check for new pyobjects after
    # the warmup, retracing each time without cleaning up old traces fails the
    # test. It works because of experimental_relax_shapes.
    execution_count = getattr(self, "_execution_count", 0)
    self._execution_count = execution_count + 1
    x = array_ops.zeros([execution_count])
    with forwardprop.ForwardGradientAccumulator() as acc:
      acc.watch(x, array_ops.ones_like(x))
      y = x + x
    self.assertAllClose(2. * array_ops.ones_like(x), acc.jvp(y))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testMultipleWatchesAdd(self):
    x = constant_op.constant(-2.)
    with forwardprop.ForwardGradientAccumulator() as acc:
      acc.watch(x, constant_op.constant(10.))
      self.assertAllClose(10., acc.jvp(x))
      acc.watch(x, constant_op.constant(11.))
      self.assertAllClose(21., acc.jvp(x))
      y = constant_op.constant(3.) * x
    self.assertAllClose(21., acc.jvp(x))
    self.assertAllClose(21. * 3., acc.jvp(y))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testDeadTensorsJVPCleared(self):
    x = array_ops.ones([100])
    x_weak = weakref.ref(x)
    grad_tensor = constant_op.constant(array_ops.zeros([100]))
    grad_tensor_weak = weakref.ref(grad_tensor)
    with forwardprop.ForwardGradientAccumulator() as acc:
      acc.watch(x, grad_tensor)
      derived_tensor = constant_op.constant(2.) * x
      del grad_tensor
      self.assertAllClose(array_ops.zeros([100]), acc.jvp(x))
      del x
      self.assertIsNone(x_weak())
      self.assertIsNone(grad_tensor_weak())
      derived_tensor_weak = weakref.ref(derived_tensor)
      derived_tensor_grad = acc.jvp(derived_tensor)
      derived_tensor_grad_weak = weakref.ref(derived_tensor_grad)
      del derived_tensor
      del derived_tensor_grad
      self.assertIsNone(derived_tensor_weak())
      self.assertIsNone(derived_tensor_grad_weak())

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testJVPManual(self):
    primal, tangent = _jvp(math_ops.sin, (constant_op.constant(0.1),),
                           (constant_op.constant(0.2),))
    self.assertAllClose(math_ops.sin(0.1), primal)
    self.assertAllClose(math_ops.cos(0.1) * 0.2, tangent)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testNumericHigherOrder(self):

    def f(x):
      pointwise = math_ops.sin(x) * math_ops.tan(x)
      return math_ops.reduce_prod(
          pointwise + math_ops.reduce_sum(pointwise), axis=1)

    _test_gradients(
        self, f, [constant_op.constant([[2.0, 3.0], [1.0, 4.0]])], order=3)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testCustomGradient(self):

    @custom_gradient.custom_gradient
    def f(x):

      def grad(dy):
        return dy * math_ops.cos(x)

      return np.sin(x.numpy()), grad

    _test_gradients(self, f, [constant_op.constant([1., 2.])], order=3)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testCustomGradientRecomputeGrad(self):

    @custom_gradient.recompute_grad
    def f(x):
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    _test_gradients(self, f, [constant_op.constant([1.])], order=3)

  def testExceptionInCustomGradientNotSwallowed(self):

    @custom_gradient.custom_gradient
    def f(unused_x):
      def grad(unused_dy):
        raise ValueError("test_error_string")
      return 1., grad

    with forwardprop.ForwardGradientAccumulator() as acc:
      c = constant_op.constant(1.)
      d = constant_op.constant(2.)
      acc.watch(c, d)
      with self.assertRaisesRegexp(ValueError, "test_error_string"):
        f(c)

  def testPushPopAccumulatorState(self):
    # Note that this example is somewhat contrived. push_forwardprop_state is
    # probably only useful in practice for building functions that compute jvps
    # alongside their usual outputs.
    with forwardprop.ForwardGradientAccumulator() as acc:

      @custom_gradient.custom_gradient
      def f(x):
        y = math_ops.sin(x.numpy())

        def grad(dy):
          with forwardprop_util.push_forwardprop_state():
            x_copy = constant_op.constant(x.numpy())
            acc.watch(x_copy, dy)
            y_copy = math_ops.sin(x_copy)
          return dy * acc.jvp(y_copy)

        return y, grad

      c = constant_op.constant(1.)
      d = constant_op.constant(2.)
      acc.watch(c, d)
      output = f(c)
      self.assertAllClose(d * math_ops.cos(c), acc.jvp(output))

  @parameterized.named_parameters(
      [("Order{}".format(order), order, expected)
       for order, expected in enumerate(_X11_35_DERIVATIVES)])
  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testHigherOrderPureForward(self, order, expected):

    def _forwardgrad(f):
      def _compute_forwardgrad(primal):
        tangent = constant_op.constant(1.)
        with forwardprop.ForwardGradientAccumulator() as acc:
          acc.watch(primal, tangent)
          primal_out = f(primal)
        return acc.jvp(primal_out)
      return _compute_forwardgrad

    def _forward(x):
      return x ** 3.5

    f = _forward
    primal = constant_op.constant(1.1)
    for _ in range(order):
      f = _forwardgrad(f)
    self.assertAllClose(expected, f(primal))

  @parameterized.named_parameters(
      [("Function", def_function.function),
       ("NoFunction", lambda f: f)])
  def testGradPureForward(self, decorator):

    @decorator
    def f(x):
      return x ** 3.5

    primal = constant_op.constant(1.1)
    with forwardprop.ForwardGradientAccumulator() as outer_acc:
      outer_acc.watch(primal, constant_op.constant(1.))
      with forwardprop.ForwardGradientAccumulator() as acc:
        acc.watch(primal, constant_op.constant(1.))
        primal_out = f(primal)
    inner_jvp = acc.jvp(primal_out)
    outer_jvp = outer_acc.jvp(inner_jvp)
    self.assertAllClose(1.1 ** 3.5, primal_out)
    self.assertAllClose(3.5 * 1.1 ** 2.5, inner_jvp)
    self.assertAllClose(3.5 * 2.5 * 1.1 ** 1.5, outer_jvp)
    self.assertIsNone(acc.jvp(outer_acc.jvp(primal_out)))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testJVPPacking(self):
    two = constant_op.constant(2.)
    with forwardprop.ForwardGradientAccumulator() as outer_acc:
      primal_in = constant_op.constant(1.)
      outer_acc.watch(primal_in, constant_op.constant(2.))
      with forwardprop.ForwardGradientAccumulator() as inner_acc:
        inner_jvp = constant_op.constant(3.)
        inner_acc.watch(primal_in, inner_jvp)
        outer_acc.watch(inner_jvp, constant_op.constant(4.))
        packed_input_indices, packed_input_tangents = (
            pywrap_tensorflow.TFE_Py_PackForwardGradients([primal_in]))
        self.assertAllClose([3., 2., 4.], packed_input_tangents)
        expected_indices = (
            # inner_acc watches primal_in
            ((0, 1),),
            # outer_acc watches primal_in and inner_jvp
            ((0, 2),
             (1, 3)))
        self.assertAllEqual(expected_indices, packed_input_indices)
        primal_out = primal_in * two
        self.assertAllClose(6., inner_acc.jvp(primal_out))
        self.assertAllClose(4., outer_acc.jvp(primal_out))
        self.assertAllClose(8., outer_acc.jvp(inner_acc.jvp(primal_out)))
        packed_output_indices, packed_output_tangents = (
            pywrap_tensorflow.TFE_Py_PackForwardGradients([primal_out]))
        self.assertAllClose([6., 4., 8.], packed_output_tangents)
        self.assertAllEqual(expected_indices, packed_output_indices)

  def testFunctionGradInFunctionPureForward(self):

    @def_function.function
    def take_gradients():
      @def_function.function
      def f(x):
        return x ** 3.5

      primal = constant_op.constant(1.1)
      with forwardprop.ForwardGradientAccumulator() as outer_acc:
        outer_acc.watch(primal, constant_op.constant(1.))
        with forwardprop.ForwardGradientAccumulator() as acc:
          acc.watch(primal, constant_op.constant(1.))
          primal_out = f(primal)
      inner_jvp = acc.jvp(primal_out)
      outer_jvp = outer_acc.jvp(inner_jvp)
      self.assertIsNone(acc.jvp(outer_acc.jvp(primal_out)))
      return primal_out, inner_jvp, outer_jvp

    primal_out, inner_jvp, outer_jvp = take_gradients()
    self.assertAllClose(1.1 ** 3.5, primal_out)
    self.assertAllClose(3.5 * 1.1 ** 2.5, inner_jvp)
    self.assertAllClose(3.5 * 2.5 * 1.1 ** 1.5, outer_jvp)

  def testFunctionGrad(self):

    @def_function.function
    def f(x):
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    _test_gradients(
        self,
        f,
        [constant_op.constant([1., 2.])],
        order=3)

  def testReusingForwardGradient(self):
    m1 = random_ops.random_uniform((256, 2096))
    m2 = array_ops.identity(m1)
    tangent1 = random_ops.random_uniform((256, 2096))
    tangent2 = random_ops.random_uniform((256, 2096))
    matmul = def_function.function(math_ops.matmul)

    with forwardprop.ForwardGradientAccumulator() as acc:
      acc.watch(m1, tangent1)
      result1 = matmul(m1, m1, transpose_b=True)
      acc.watch(m2, tangent2)
      result2 = matmul(m2, m2, transpose_b=True)

    def _expected(mat, tangent):
      return (math_ops.matmul(tangent, mat, transpose_b=True)
              + math_ops.matmul(mat, tangent, transpose_b=True))

    self.assertAllClose(result1, result2)
    self.assertAllClose(_expected(m1, tangent1), acc.jvp(result1))
    self.assertAllClose(_expected(m2, tangent2), acc.jvp(result2))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testHVPMemory(self):

    def fun(x):
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    primals = constant_op.constant([1., 2., 3.])
    tangents = constant_op.constant([3., 4., 5.])
    _hvp(fun, (primals,), (tangents,))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testHVPCorrectness(self):

    def fun(x):
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    primals = constant_op.constant([1., 2., 3.])
    tangents = constant_op.constant([3., 4., 5.])
    forwardback_hvp_eager = _hvp(fun, (primals,), (tangents,))
    forwardback_hvp_function = def_function.function(_hvp)(fun, (primals,),
                                                           (tangents,))

    with backprop.GradientTape(persistent=True) as g:
      g.watch(primals)
      with backprop.GradientTape() as gg:
        gg.watch(primals)
        out = fun(primals)
      grad = array_ops.unstack(gg.gradient(out, primals))
    hessian = []
    for i in range(3):
      hessian.append(g.gradient(grad[i], primals))
    hessian = array_ops.stack(hessian, axis=0)
    backback_hvp = math_ops.tensordot(hessian, tangents, axes=1)

    self.assertAllClose(backback_hvp, forwardback_hvp_eager)
    self.assertAllClose(backback_hvp, forwardback_hvp_function)


if __name__ == "__main__":
  # TODO(allenl): Also test with 1.x-style graph mode.
  ops.enable_eager_execution()
  test.main()
