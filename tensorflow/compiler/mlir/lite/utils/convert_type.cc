/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"

#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

mlir::Type ConvertElementType(tflite::TensorType type, mlir::Builder builder) {
  switch (type) {
    case tflite::TensorType_FLOAT32:
      return builder.getF32Type();
    case tflite::TensorType_FLOAT16:
      return builder.getF16Type();
    case tflite::TensorType_INT32:
      return builder.getIntegerType(32);
    case tflite::TensorType_UINT8:
      return mlir::TF::Uint8Type::get(builder.getContext());
    case tflite::TensorType_INT64:
      return builder.getIntegerType(64);
    case tflite::TensorType_STRING:
      return mlir::TF::StringType::get(builder.getContext());
    case tflite::TensorType_BOOL:
      return builder.getI1Type();
    case tflite::TensorType_INT16:
      return builder.getIntegerType(16);
    case tflite::TensorType_COMPLEX64:
      return mlir::TF::Complex64Type::get(builder.getContext());
    case tflite::TensorType_INT8:
      return builder.getIntegerType(8);
  }
}

tensorflow::DataType TflTypeToTfType(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_BOOL:
      return tensorflow::DT_BOOL;
    case tflite::TensorType_COMPLEX64:
      return tensorflow::DT_COMPLEX64;
    case tflite::TensorType_FLOAT16:
      return tensorflow::DT_HALF;
    case tflite::TensorType_FLOAT32:
      return tensorflow::DT_FLOAT;
    case tflite::TensorType_INT8:
      return tensorflow::DT_INT8;
    case tflite::TensorType_INT16:
      return tensorflow::DT_INT16;
    case tflite::TensorType_INT32:
      return tensorflow::DT_INT32;
    case tflite::TensorType_INT64:
      return tensorflow::DT_INT64;
    case tflite::TensorType_STRING:
      return tensorflow::DT_STRING;
    case tflite::TensorType_UINT8:
      return tensorflow::DT_UINT8;
  }
}

}  // namespace tflite
