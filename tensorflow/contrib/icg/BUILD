# Description:
# Custom operators for ICGVN module.

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_library(
    name = "python/ops/_icg_ops.so",
    srcs = ["ops/ffttools.cc",
            "common/definitions.h",
            "ops/activations.h", 
            "ops/activations.cc",
            ],
    gpu_srcs = ["kernels/ffttools.cu.cc",
                "common/definitions.cu.cc",
                "common/definitions.h",
                "ops/activations.h",
                "kernels/activations.cu.cc",
                ],
)

tf_kernel_library(
    name = "icg_ops_kernels",
    srcs = ["ops/ffttools.cc",
            "common/definitions.h",
            "ops/activations.h", 
            "ops/activations.cc",
           ],
    gpu_srcs = ["kernels/ffttools.cu.cc",
                "common/definitions.cu.cc",
                "common/definitions.h",
                "ops/activations.h",
                "kernels/activations.cu.cc",
               ],
    deps = [
                  "//tensorflow/core:framework",
                  "//tensorflow/core:lib",
                  "//third_party/eigen3",
              ],
    alwayslink = 1,
)

tf_custom_op_py_library(
    name = "icg_py",
    srcs = [
        "__init__.py",
        "python/ops/icg_ops.py",
        "python/variationalnetwork.py",
        "python/utils.py",
        "python/optimizer.py"
    ],
    dso = [":python/ops/_icg_ops.so"],
    kernels = [
        ":icg_ops_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)


filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
