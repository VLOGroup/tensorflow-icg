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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_

#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"

namespace tensorflow {

// Quantization passess will run only when the user specifies a quantized type
// in the `-tf-inference-type` flag, which is converted to the function
// attribute "tf.quantize" by the importer module.
// TODO(fengliuai): switch to the cmd flag once the flags are moved to this
// file with main method.
bool ShouldRunQuantizePasses(mlir::ModuleOp m);

// Add the TF to TFLite passes, specified in the pass_config, into a
// pass_manager.
void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::PassManager* pass_manager);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_
