// RUN: tf-opt %s -verify-diagnostics -split-input-file

// -----

func @enforce_static_shapes(%arg0: memref<?xf32>, %arg1: memref<?xf32>) -> () {
  // expected-error@+1{{op operand #0 must be statically shaped memref of floating-point or integer values}}
  "xla_lhlo.tanh"(%arg0, %arg1) : (memref<?xf32>, memref<?xf32>) -> ()
  return
}

// -----

func @enforce_same_shape(%arg0: memref<1xf32>, %arg1: memref<2xf32>) -> () {
  // expected-error@+1{{'xla_lhlo.tanh' op requires all operands to have the same type}}
  "xla_lhlo.tanh"(%arg0, %arg1) : (memref<1xf32>, memref<2xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @add_memrefs
func @add_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "xla_lhlo.add"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @abs_memref
func @abs_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.abs"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @convert_memref
func @convert_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.convert"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @exp_memref
func @exp_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.exp"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @neg_memref
func @neg_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.neg"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sign_memref
func @sign_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.sign"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @tanh_memref
func @tanh_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.tanh"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @add_memref
func @add_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.add"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @div_memref
func @div_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.div"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @max_memref
func @max_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.max"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @min_memref
func @min_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.min"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @mul_memref
func @mul_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.mul"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sub_memref
func @sub_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.sub"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @and_memref
func @and_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.and"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

func @reduce_computation(%sum: memref<1xf32>, %element: memref<1xf32>) -> () {
  "xla_lhlo.add"(%element, %sum, %sum) : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  return
}

// CHECK-LABEL: func @reduce_memref
func @reduce_memref(%input: memref<10xf32>, %out: memref<1xf32>) -> () {
  "xla_lhlo.reduce"(%input, %out) {computation = @reduce_computation,
                                   dimensions = dense<[0]> : tensor<1xi64>} : (memref<10xf32>, memref<1xf32>) -> ()
  return
}