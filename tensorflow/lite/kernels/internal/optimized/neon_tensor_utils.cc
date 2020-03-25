/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/experimental/ruy/detect_arm.h"
#include "tensorflow/lite/kernels/activation_functor.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_tensor_utils_impl.h"
#include "tensorflow/lite/kernels/internal/round.h"

#ifdef USE_NEON

// aligned_alloc is available (via cstdlib/stdlib.h) with C++17/C11.
#if __cplusplus >= 201703L || __STDC_VERSION__ >= 201112L
#if !defined(__ANDROID__) || __ANDROID_API__ >= 28
#if !defined(__APPLE__)  // Apple does not provide aligned_alloc.
#define TFLITE_USE_STD_ALIGNED_ALLOC
#endif
#endif
#endif

namespace tflite {
namespace tensor_utils {
namespace {

constexpr int kFloatValuesPerNeonVector = 4;

inline int RoundDownToFloatVectors(int size) {
  return size & ~(kFloatValuesPerNeonVector - 1);
}

// Allocates, at least, size bytes of uninitialized storage whose alignment is
// specified by alignment. The size parameter must be an integral multiple of
// alignment.
// Caller is responsible by freeing the allocated memory by calling free on
// the passed freeing_buffer pointer.
inline void* aligned_alloc(size_t alignment, size_t size,
                           void** freeing_buffer) {
#ifdef TFLITE_USE_STD_ALIGNED_ALLOC
  *freeing_buffer = ::aligned_alloc(
      alignment, (size + alignment - 1) / alignment * alignment);
  return *freeing_buffer;
#else
  *freeing_buffer = malloc(size + alignment);
  const size_t offset = ((uintptr_t)*freeing_buffer) % alignment;  // NOLINT
  return offset == 0
             ? *freeing_buffer
             : ((char*)*freeing_buffer + (alignment - offset));  // NOLINT
#endif
}

bool HasSdotInstruction() {
  static const bool has_dotprod = ruy::DetectDotprod();
  return has_dotprod;
}

inline float AccumulateNeonLane(const float32x4_t lane) {
#ifdef __aarch64__
  return vaddvq_f32(lane);
#else
  return vgetq_lane_f32(lane, 0) + vgetq_lane_f32(lane, 1) +
         vgetq_lane_f32(lane, 2) + vgetq_lane_f32(lane, 3);
#endif
}

inline int32_t AccumulateNeonLane(const int32x4_t lane) {
#ifdef __aarch64__
  return vaddvq_s32(lane);
#else
  int64x2_t pairwiseAdded = vpaddlq_s32(lane);
  return vgetq_lane_s64(pairwiseAdded, 0) + vgetq_lane_s64(pairwiseAdded, 1);
#endif
}

inline int64_t AccumulateNeonLane64(const int64x2_t lane) {
#ifdef __aarch64__
  return vaddvq_s64(lane);
#else
  return vgetq_lane_s64(lane, 0) + vgetq_lane_s64(lane, 1);
#endif
}

}  // namespace

void NeonMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride) {
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownToFloatVectors(m_cols);

  for (int b = 0; b < n_batch; b++) {
    float* result_in_batch = result + b * m_rows * result_stride;
    const float* vector_in_batch = vector + b * m_cols;
    const float* matrix_row = matrix;

    // Main matrix by vector multiplication loop
    for (int r = 0; r < m_rows; r++) {
      float32x4_t acc_32x4 = vmovq_n_f32(0.0);
      int c = 0;
      for (; c < postamble_start; c += kFloatValuesPerNeonVector) {
        // Load 4 float values from vector and matrix row.
        float32x4_t vector_f32x4 = vld1q_f32(vector_in_batch + c);
        float32x4_t matrix_f32x4 = vld1q_f32(matrix_row + c);
        // Multiply the vector and matrix row and add to accumulator.
        acc_32x4 = vmlaq_f32(acc_32x4, matrix_f32x4, vector_f32x4);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this column.
      *result_in_batch += AccumulateNeonLane(acc_32x4);
      for (; c < m_cols; c++) {
        *result_in_batch += matrix_row[c] * vector_in_batch[c];
      }
      matrix_row += m_cols;
      result_in_batch += result_stride;
    }
  }
}

#ifdef __aarch64__

// We interleave vector data to make the dot product logic more efficient.
// Suppose that vectors is:
//     a0 a1 a2 a3 a4 a5 ...
//     b0 b1 b2 b3 b4 b5 ...
//     c0 c1 c2 c3 c4 c5 ...
//     d0 d1 d2 d3 d4 d5 ...
//     e0 e1 e2 e3 e4 e5 ...
// This code interleaves them like this:
//     a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3 d0 d1 d2 d3 a4 a5 a6 a7 b4 ...
//     e0 e1 e2 e3 f0 f1 f2 f3 ...
// Once the data is interleaved, each 16-byte read from the vectors pointer
// contains 4 bytes from each of 4 vectors.
const int8_t* ShuffleVectors(const int8_t* vectors, const int n_batch,
                             const int m_cols, void** shuffled_vectors_free) {
  const int kWeightsPerUint32 = 4;

  int8* shuffled_vectors = reinterpret_cast<int8*>(aligned_alloc(
      kWeightsPerUint32, n_batch * m_cols, shuffled_vectors_free));

  for (int i = 0; i < n_batch; i += 4) {
    int8* shuffled_vectors_ptr = shuffled_vectors + (i * m_cols);
    const int8* unshuffled_vec0_ptr =
        reinterpret_cast<const int8*>(vectors) + (i * m_cols);
    const int8* unshuffled_vec1_ptr =
        reinterpret_cast<const int8*>(vectors) + ((i + 1) * m_cols);
    const int8* unshuffled_vec2_ptr =
        reinterpret_cast<const int8*>(vectors) + ((i + 2) * m_cols);
    const int8* unshuffled_vec3_ptr =
        reinterpret_cast<const int8*>(vectors) + ((i + 3) * m_cols);
    const int8* const end_vec0_ptr = unshuffled_vec1_ptr;

    while (unshuffled_vec0_ptr != end_vec0_ptr) {
      asm volatile(
          // This code path requires that (n_cols % 16) == 0 so we can safely
          // read in 16-byte chunks from each row.
          "ld1 {v0.16b}, [%[unshuffled_vec0_ptr]], #16\n"
          "ld1 {v1.16b}, [%[unshuffled_vec1_ptr]], #16\n"
          "ld1 {v2.16b}, [%[unshuffled_vec2_ptr]], #16\n"
          "ld1 {v3.16b}, [%[unshuffled_vec3_ptr]], #16\n"

          "st4 {v0.s, v1.s, v2.s, v3.s}[0], [%[shuffled_vectors_ptr]], #16\n"
          "st4 {v0.s, v1.s, v2.s, v3.s}[1], [%[shuffled_vectors_ptr]], #16\n"
          "st4 {v0.s, v1.s, v2.s, v3.s}[2], [%[shuffled_vectors_ptr]], #16\n"
          "st4 {v0.s, v1.s, v2.s, v3.s}[3], [%[shuffled_vectors_ptr]], #16\n"

          : [ unshuffled_vec0_ptr ] "+r"(unshuffled_vec0_ptr),
            [ unshuffled_vec1_ptr ] "+r"(unshuffled_vec1_ptr),
            [ unshuffled_vec2_ptr ] "+r"(unshuffled_vec2_ptr),
            [ unshuffled_vec3_ptr ] "+r"(unshuffled_vec3_ptr),
            [ shuffled_vectors_ptr ] "+r"(shuffled_vectors_ptr)
          :
          : "v0", "v1", "v2", "v3", "cc", "memory");
    }
  }

  return reinterpret_cast<const int8_t*>(shuffled_vectors);
}

static void DotprodMatrixBatchFourVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* vectors, const float* scaling_factors, int n_batch,
    float* __restrict__ result) {
  void* shuffled_vectors_free;

  const int8_t* shuffled_vectors =
      ShuffleVectors(vectors, n_batch, m_cols, &shuffled_vectors_free);

  for (int row = 0; row < m_rows; row += 2) {
    for (int batch = 0; batch < n_batch; batch += 4) {
      float* result_ptr = result + (batch * m_rows) + row;
      const int8* mat_ptr0 = matrix + (row * m_cols);
      const int8* mat_ptr1 = matrix + ((row + 1) * m_cols);
      const int8* mat_ptr0_end = mat_ptr1;
      const int8* vec_ptr = shuffled_vectors + (batch * m_cols);
      const float* scaling_factors_ptr = scaling_factors + batch;
      const uint64_t wide_rows = m_rows * sizeof(float);

      asm volatile(
          // Zero out the accumulator registers.
          "dup v0.4s, wzr\n"
          "dup v1.4s, wzr\n"
          "dup v2.4s, wzr\n"
          "dup v3.4s, wzr\n"

          "1:\n"  // batch_cols_loop

          // Read 16 more bytes from a pair of matrix rows.
          "ld1 {v12.16b}, [%[mat_ptr0]], #16\n"

          // Read from input vectors 4 times; 64 bytes total.
          // Each 16-byte register contains parts of 4 vectors; see the
          // shuffle logic above.

          // From Benoit, places to look in the future:
          // - Move load instructions further from sdot
          // - Switch loop use-then-reload
          // - Do partial unrolling to use register space better
          "ld1 {v8.16b}, [%[vec_ptr]], #16\n"
          ".word 0x4f8ce100  // sdot v0.4s, v8.16b, v12.4b[0]\n"
          "ld1 {v9.16b}, [%[vec_ptr]], #16\n"
          ".word 0x4face121  // sdot v1.4s, v9.16b, v12.4b[1]\n"
          "ld1 {v10.16b}, [%[vec_ptr]], #16\n"
          ".word 0x4f8ce940  // sdot v0.4s, v10.16b, v12.4b[2]\n"
          "ld1 {v11.16b}, [%[vec_ptr]], #16\n"
          ".word 0x4face961  // sdot v1.4s, v11.16b, v12.4b[3]\n"

          // Re-use those vectors for the next row as well.
          "ld1 {v13.16b}, [%[mat_ptr1]], #16\n"
          ".word 0x4f8de102  // sdot v2.4s, v8.16b, v13.4b[0]\n"
          ".word 0x4fade123  // sdot v3.4s, v9.16b, v13.4b[1]\n"
          ".word 0x4f8de942  // sdot v2.4s, v10.16b, v13.4b[2]\n"
          ".word 0x4fade963  // sdot v3.4s, v11.16b, v13.4b[3]\n"

          // If we're not done with these rows, continue.
          "cmp %[mat_ptr0], %[mat_ptr0_end]\n"
          "bne 1b\n"  // batch_cols_loop

          // Done with the rows, sum the results.
          "add v0.4s, v0.4s, v1.4s\n"
          "add v2.4s, v2.4s, v3.4s\n"

          // Convert the per-vector sums to floating point.
          "scvtf v0.4s, v0.4s\n"
          "scvtf v1.4s, v2.4s\n"

          // Fetch scale factors.
          "ld1 {v4.4s}, [%[scaling_factors_ptr]]\n"

          // Multiply scale factors times sums.
          "fmul v0.4s, v4.4s, v0.4s\n"
          "fmul v1.4s, v4.4s, v1.4s\n"

          // Load previous result values.
          // The result position is:
          //   result[batch * m_rows + row]
          // Here that is factored into:
          //   result_ptr = result + row
          //   *result_ptr = res[0]
          //   (uint8*)result_ptr += (m_rows * sizeof(float))
          //   *result_ptr = res[1]
          //   ...
          // Since we're reading two rows at a time, though, we read both
          //   result[batch * m_rows + row]
          // and
          //   result[batch * m_rows + row + 1]
          "ld2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
          "ld2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
          "ld2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
          "ld2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"

          // Go back to the starting position (subtract wide_rows * 4).
          "sub %[result_ptr], %[result_ptr], %[wide_rows], lsl #2\n"

          // Add previous result values.
          "fadd v9.4s, v9.4s, v0.4s\n"
          "fadd v10.4s, v10.4s, v1.4s\n"

          // Store results.
          "st2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
          "st2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
          "st2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
          "st2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"
          : [ mat_ptr0 ] "+r"(mat_ptr0), [ mat_ptr1 ] "+r"(mat_ptr1),
            [ vec_ptr ] "+r"(vec_ptr), [ result_ptr ] "+r"(result_ptr)
          : [ mat_ptr0_end ] "r"(mat_ptr0_end),
            [ scaling_factors_ptr ] "r"(scaling_factors_ptr),
            [ wide_rows ] "r"(wide_rows)
          : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
            "v10", "v11", "v12", "v13", "cc", "memory");
    }
  }

  free(shuffled_vectors_free);
}

static void DotprodSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    int result_stride) {
  const uint8_t* ledger_ptr = ledger;
  const int8* mat_ptr = matrix;

  for (int row = 0; row < m_rows; row++) {
    int num_nonzero_chunks = *ledger_ptr;
    ledger_ptr++;
    const uint8* ledger_start = ledger_ptr;
    const uint8* ledger_end = ledger_ptr + num_nonzero_chunks;
    const int8* mat_start = mat_ptr;

    for (int batch = 0; batch < n_batch; batch++) {
      const int8* vec_ptr = vectors + (batch * m_cols);
      int64_t row_sum = 0;

      mat_ptr = mat_start;
      ledger_ptr = ledger_start;

      if (ledger_ptr != ledger_end) {
        asm volatile(
            "dup v0.4s, wzr\n"
            "dup v1.4s, wzr\n"
            "dup v8.4s, wzr\n"
            "mov x7, 0\n"

            "1:\n"  // chunks_loop

            // Single matrix chunk, 16 bytes
            "ld1 {v8.16b}, [%[mat_ptr]], #16\n"

            // Read the next ledger index and increment.
            "ldrb w7, [%[ledger_ptr]], #1\n"

            // Read 16 bytes of vector data from (vec_ptr + (ledger_index * 16))
            "add x8, %[vec_ptr], x7, lsl #4\n"
            "ld1 {v9.16b}, [x8]\n"

            // Dot product of matrix row and vector.
            ".word 0x4e889520  // sdot v0.4s, v9.16b, v8.16b\n"

            "cmp %[ledger_ptr], %[ledger_end]\n"
            "blt 1b\n"  // chunks_loop

            // Sum the 4 vector components into a 32-bit value.
            "addv s1, v0.4s\n"
            // row_sum is 64-bit, so we copy 64 bits of v1 into it.
            // We have to be careful to cast this value to 32 bits in order
            // to interpret the sign bit properly.
            "mov %[row_sum], v1.d[0]\n"
            : [ row_sum ] "=r"(row_sum), [ ledger_ptr ] "+r"(ledger_ptr),
              [ mat_ptr ] "+r"(mat_ptr), [ vec_ptr ] "+r"(vec_ptr)
            : [ ledger_end ] "r"(ledger_end)
            : "x0", "x1", "x7", "x8", "v0", "v1", "v8", "v9", "cc", "memory");
      }
      result[(batch * m_rows + row) * result_stride] +=
          static_cast<int32>(row_sum) * scaling_factors[batch];
    }
  }
}

#endif  // __aarch64__

void NeonMatrixBatchVectorMultiplyImpl(
    const int8_t* input, const int32_t* input_zeropoint_times_weights,
    const int8_t* input_to_gate_weights, int32_t n_batch, int32_t n_input,
    int32_t n_output, int32_t output_zp, int32_t* scratch) {
  static const int kWeightsPerUint32 = 4;
  static const int kWeightsPerNeonLane = 16;
  // Assuming *matrix is kWeightsPerUint32-byte aligned,
  // every row of the matrix is also
  // kWeightsPerUint32-byte aligned as long as cols is
  // a multiple of kWeightsPerUint32. The assumption
  // is currently satisfied by TFLite's 16-byte memory
  // alignment scheme.
  //
  // Otherwise, we allocate an aligned memory block and set
  // a flag to later copy rows from matrix to the block
  // for aligned multiplication.
  bool unaligned = false;
  int8_t* aligned_row = nullptr;
  void* aligned_row_free = nullptr;
  if ((n_input & (kWeightsPerUint32 - 1)) != 0) {
    unaligned = true;
    aligned_row = (int8_t*)aligned_alloc(kWeightsPerUint32, n_input,  // NOLINT
                                         &aligned_row_free);
  }
  void* aligned_vec_free = nullptr;
  int8_t* aligned_vec =
      (int8_t*)aligned_alloc(kWeightsPerUint32, n_input,  // NOLINT
                             &aligned_vec_free);

  // If m_cols is not at least kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_half_start
  // shows the start index where this should happen. Between postamble_start and
  // postamble_half_start we can still process kWeightsPerNeonLane >> 1 in a
  // vectorized form.
  const int postamble_half_start = n_input & ~(kWeightsPerNeonLane - 1);
  const int postamble_start = n_input & ~((kWeightsPerNeonLane >> 1) - 1);

  for (int batch = 0; batch < n_batch; ++batch) {
    // Copy the vector data to an aligned vector.
    memcpy(aligned_vec, input + batch * n_input, sizeof(int8_t) * n_input);
    // Compute dot-product for every column.
    for (int row = 0; row < n_output; ++row) {
      // Get the address of the first element of the row.
      int8_t* row_ptr =
          (int8_t*)input_to_gate_weights + row * n_input;  // NOLINT
      if (unaligned) {
        memcpy(aligned_row, row_ptr, sizeof(int8_t) * n_input);
        row_ptr = aligned_row;
      }

      // Initialize the dot product sum for the row to 0.
      int32x4_t dotprod_32x4 = vmovq_n_s32(0);

      // For every block of 16 8-bit elements.
      int col = 0;
      for (; col < postamble_half_start; col += kWeightsPerNeonLane) {
        // Load 16 8-bit values from the row and vector, each, to operate on.
        // Here the assumption is that each buffer is 4-byte aligned. Otherwise,
        // performance may suffer significantly.
        TFLITE_DCHECK_EQ(  // NOLINT
            (uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1), 0);
        const int8x16_t s1_8x16 = vld1q_s8((const int8_t*)(aligned_vec + col));
        const int8x16_t s2_8x16 = vld1q_s8((const int8_t*)(row_ptr + col));
        // Multiply the low bits (i.e. the lower 8 8bit numbers in the
        // registers).
        int16x8_t prod_16x8 =
            vmull_s8(vget_low_s8(s1_8x16), vget_low_s8(s2_8x16));
        // Multiply the high bits (i.e. the higher 8 8bit numbers in the
        // registers), and accumulate with the result of the low bits product.
        // The assumption here is that overflow will not happen as we quantize
        // our values to be in the range [-127, 127]. As such the sum of the 2
        // products is always strictly smaller than 15-bits (32767 in absolute
        // value).
        prod_16x8 =
            vmlal_s8(prod_16x8, vget_high_s8(s1_8x16), vget_high_s8(s2_8x16));

        dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8);
      }  // for col

      // Half iteration dealing only 8 elements
      // TODO(raziel): if (ABSL_PREDICT_FALSE(col < postamble_start))
      if (col < postamble_start) {
        // Load 8 8-bit values from the row and column each to operate on.
        // Here the assumption is that each buffer is 4-bytes aligned.
        // Otherwise, performance may suffer significantly.
        TFLITE_DCHECK_EQ(  // NOLINT
            (uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1), 0);
        const int8x8_t s1_8x8 = vld1_s8((const int8_t*)(aligned_vec + col));
        const int8x8_t s2_8x8 = vld1_s8((const int8_t*)(row_ptr + col));
        const int16x8_t prod_16x8 = vmull_s8(s1_8x8, s2_8x8);
        dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8);
        col += (kWeightsPerNeonLane >> 1);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this row.
      int32_t dotprod = AccumulateNeonLane(dotprod_32x4);
      // Postamble loop.
      // TODO(raziel): if (ABSL_PREDICT_FALSE(col < m_cols))
      for (; col < n_input; ++col) {
        dotprod += row_ptr[col] * aligned_vec[col];
      }  // for col

      dotprod += input_zeropoint_times_weights[row];
      scratch[batch * n_output + row] = dotprod;
    }  // for row
  }    // for batch

  if (unaligned) {
    free(aligned_row_free);
  }
  free(aligned_vec_free);
}

void NeonMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* input_zeropoint_times_weights,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output) {
  NeonMatrixBatchVectorMultiplyImpl(input, input_zeropoint_times_weights,
                                    input_to_gate_weights, n_batch, n_input,
                                    n_output, output_zp, scratch);
  int i = 0;
  const int total_size = n_batch * n_output;

  const int32_t output_min = std::numeric_limits<int16_t>::min();
  const int32_t output_max = std::numeric_limits<int16_t>::max();

  const int32_t left_shift_one = shift > 0 ? 1 << shift : 1;
  const int32_t right_shift = shift > 0 ? 0 : -shift;

  const int32x4_t left_shift_one_dup = vdupq_n_s32(left_shift_one);
  const int32x4_t output_zp_dup = vdupq_n_s32(output_zp);
  const int32x4_t max_val_dup = vdupq_n_s32(output_max);
  const int32x4_t min_val_dup = vdupq_n_s32(output_min);

  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;

  for (; i <= total_size - 8; i += 8) {
    int32x4x2_t scratch_val;
    scratch_val.val[0] = vld1q_s32(scratch + i);
    scratch_val.val[1] = vld1q_s32(scratch + i + 4);
    const int16x8_t output_val = vld1q_s16(output + i);
    const int32x4_t first_half = vmovl_s16(vget_low_s16(output_val));
    const int32x4_t second_half = vmovl_s16(vget_high_s16(output_val));
    int32x4_t temp_val_1 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(
            vmulq_s32(scratch_val.val[0], left_shift_one_dup), multiplier),
        right_shift);
    int32x4_t temp_val_2 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(
            vmulq_s32(scratch_val.val[1], left_shift_one_dup), multiplier),
        right_shift);
    temp_val_1 = vaddq_s32(vaddq_s32(temp_val_1, first_half), output_zp_dup);
    temp_val_2 = vaddq_s32(vaddq_s32(temp_val_2, second_half), output_zp_dup);
    temp_val_1 = vmaxq_s32(vminq_s32(temp_val_1, max_val_dup), min_val_dup);
    temp_val_2 = vmaxq_s32(vminq_s32(temp_val_2, max_val_dup), min_val_dup);
    const int16x8_t result =
        vcombine_s16(vqmovn_s32(temp_val_1), vqmovn_s32(temp_val_2));
    vst1q_s16(output + i, result);
  }
  for (; i < total_size; ++i) {
    int32_t temp = MultiplyByQuantizedMultiplier(scratch[i], multiplier, shift);
    temp += output_zp;
    temp += output[i];
    if (temp > output_max) {
      temp = output_max;
    }
    if (temp < output_min) {
      temp = output_min;
    }
    output[i] = static_cast<int16_t>(temp);
  }
}

void NeonMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* input_zeropoint_times_weights,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output) {
  NeonMatrixBatchVectorMultiplyImpl(input, input_zeropoint_times_weights,
                                    input_to_gate_weights, n_batch, n_input,
                                    n_output, output_zp, scratch);
  int i = 0;
  const int total_size = n_batch * n_output;

  const int32_t output_min = std::numeric_limits<int8_t>::min();
  const int32_t output_max = std::numeric_limits<int8_t>::max();

  const int32_t left_shift_one = shift > 0 ? 1 << shift : 1;
  const int32_t right_shift = shift > 0 ? 0 : -shift;

  const int32x4_t left_shift_one_dup = vdupq_n_s32(left_shift_one);
  const int32x4_t output_zp_dup = vdupq_n_s32(output_zp);
  const int32x4_t max_val_dup = vdupq_n_s32(output_max);
  const int32x4_t min_val_dup = vdupq_n_s32(output_min);

  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;

  for (; i <= total_size - 16; i += 16) {
    int32x4x4_t scratch_val;
    scratch_val.val[0] = vld1q_s32(scratch + i);
    scratch_val.val[1] = vld1q_s32(scratch + i + 4);
    scratch_val.val[2] = vld1q_s32(scratch + i + 8);
    scratch_val.val[3] = vld1q_s32(scratch + i + 12);

    const int8x16_t output_val = vld1q_s8(output + i);
    const int16x8_t first_half = vmovl_s8(vget_low_s8(output_val));
    const int16x8_t second_half = vmovl_s8(vget_high_s8(output_val));
    const int32x4_t output_val_1 = vmovl_s16(vget_low_s16(first_half));
    const int32x4_t output_val_2 = vmovl_s16(vget_high_s16(first_half));
    const int32x4_t output_val_3 = vmovl_s16(vget_low_s16(second_half));
    const int32x4_t output_val_4 = vmovl_s16(vget_high_s16(second_half));

    int32x4_t temp_val_1 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(
            vmulq_s32(scratch_val.val[0], left_shift_one_dup), multiplier),
        right_shift);
    int32x4_t temp_val_2 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(
            vmulq_s32(scratch_val.val[1], left_shift_one_dup), multiplier),
        right_shift);
    int32x4_t temp_val_3 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(
            vmulq_s32(scratch_val.val[2], left_shift_one_dup), multiplier),
        right_shift);
    int32x4_t temp_val_4 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(
            vmulq_s32(scratch_val.val[3], left_shift_one_dup), multiplier),
        right_shift);

    temp_val_1 = vaddq_s32(vaddq_s32(temp_val_1, output_val_1), output_zp_dup);
    temp_val_2 = vaddq_s32(vaddq_s32(temp_val_2, output_val_2), output_zp_dup);
    temp_val_3 = vaddq_s32(vaddq_s32(temp_val_3, output_val_3), output_zp_dup);
    temp_val_4 = vaddq_s32(vaddq_s32(temp_val_4, output_val_4), output_zp_dup);

    temp_val_1 = vmaxq_s32(vminq_s32(temp_val_1, max_val_dup), min_val_dup);
    temp_val_2 = vmaxq_s32(vminq_s32(temp_val_2, max_val_dup), min_val_dup);
    temp_val_3 = vmaxq_s32(vminq_s32(temp_val_3, max_val_dup), min_val_dup);
    temp_val_4 = vmaxq_s32(vminq_s32(temp_val_4, max_val_dup), min_val_dup);

    const int16x8_t result_1 =
        vcombine_s16(vqmovn_s32(temp_val_1), vqmovn_s32(temp_val_2));
    const int16x8_t result_2 =
        vcombine_s16(vqmovn_s32(temp_val_3), vqmovn_s32(temp_val_4));
    const int8x16_t result =
        vcombine_s8(vqmovn_s16(result_1), vqmovn_s16(result_2));
    vst1q_s8(output + i, result);
  }
  for (; i < total_size; ++i) {
    int32_t temp = MultiplyByQuantizedMultiplier(scratch[i], multiplier, shift);
    temp += output_zp;
    temp += output[i];
    if (temp > output_max) {
      temp = output_max;
    }
    if (temp < output_min) {
      temp = output_min;
    }
    output[i] = static_cast<int8_t>(temp);
  }
}

void NeonMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride) {
#ifdef __aarch64__
  if (HasSdotInstruction() && m_cols % 16 == 0 && m_rows % 2 == 0 &&
      m_rows >= n_batch) {
    if (n_batch % 4 == 0 && result_stride == 1) {
      // Benchmarks suggest that it's always better to use the batch code
      // when we can, even on small matrices.
      DotprodMatrixBatchFourVectorMultiplyAccumulate(
          matrix, m_rows, m_cols, vectors, scaling_factors, n_batch, result);
      return;
    }
  }
#endif  // __aarch64__

  static const int kWeightsPerUint32 = 4;
  static const int kWeightsPerNeonLane = 16;
  // Assuming *matrix is kWeightsPerUint32-byte aligned,
  // every row of the matrix is also
  // kWeightsPerUint32-byte aligned as long as cols is
  // a multiple of kWeightsPerUint32. The assumption
  // is currently satisfied by TFLite's 16-byte memory
  // alignment scheme.
  //
  // Otherwise, we allocate an aligned memory block and set
  // a flag to later copy rows from matrix to the block
  // for aligned multiplication.
  bool unaligned = false;
  int8_t* aligned_row = nullptr;
  void* aligned_row_free = nullptr;
  if ((m_cols & (kWeightsPerUint32 - 1)) != 0) {
    unaligned = true;
    aligned_row = (int8_t*)aligned_alloc(kWeightsPerUint32, m_cols,  // NOLINT
                                         &aligned_row_free);
  }
  void* aligned_vec_free = nullptr;
  int8_t* aligned_vec =
      (int8_t*)aligned_alloc(kWeightsPerUint32, m_cols,  // NOLINT
                             &aligned_vec_free);

  // If m_cols is not at least kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_half_start
  // shows the start index where this should happen. Between postamble_start and
  // postamble_half_start we can still process kWeightsPerNeonLane >> 1 in a
  // vectorized form.
  const int postamble_half_start = m_cols & ~(kWeightsPerNeonLane - 1);
  const int postamble_start = m_cols & ~((kWeightsPerNeonLane >> 1) - 1);

  for (int batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    // Copy the vector data to an aligned vector.
    memcpy(aligned_vec, vectors + batch * m_cols, sizeof(int8_t) * m_cols);
    // Compute dot-product for every column.
    for (int row = 0; row < m_rows; ++row, result += result_stride) {
      // Get the address of the first element of the row.
      int8_t* row_ptr = (int8_t*)matrix + row * m_cols;  // NOLINT
      if (unaligned) {
        memcpy(aligned_row, row_ptr, sizeof(int8_t) * m_cols);
        row_ptr = aligned_row;
      }

      // Initialize the dot product sum for the row to 0.
      int32x4_t dotprod_32x4 = vmovq_n_s32(0);

      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                         3 /* temporal locality */);

      // For every block of 16 8-bit elements.
      int col = 0;
      for (; col < postamble_half_start; col += kWeightsPerNeonLane) {
        // Load 16 8-bit values from the row and vector, each, to operate on.
        // Here the assumption is that each buffer is 4-byte aligned. Otherwise,
        // performance may suffer significantly.
        TFLITE_DCHECK_EQ(  // NOLINT
            (uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1), 0);
        const int8x16_t s1_8x16 = vld1q_s8((const int8_t*)(aligned_vec + col));
        const int8x16_t s2_8x16 = vld1q_s8((const int8_t*)(row_ptr + col));
        // Multiply the low bits (i.e. the lower 8 8bit numbers in the
        // registers).
        int16x8_t prod_16x8 =
            vmull_s8(vget_low_s8(s1_8x16), vget_low_s8(s2_8x16));
        // Multiply the high bits (i.e. the higher 8 8bit numbers in the
        // registers), and accumulate with the result of the low bits product.
        // The assumption here is that overflow will not happen as we quantize
        // our values to be in the range [-127, 127]. As such the sum of the 2
        // products is always strictly smaller than 15-bits (32767 in absolute
        // value).
        prod_16x8 =
            vmlal_s8(prod_16x8, vget_high_s8(s1_8x16), vget_high_s8(s2_8x16));

        dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8);
      }  // for col

      // Half iteration dealing only 8 elements
      // TODO(raziel): if (ABSL_PREDICT_FALSE(col < postamble_start))
      if (col < postamble_start) {
        // Load 8 8-bit values from the row and column each to operate on.
        // Here the assumption is that each buffer is 4-bytes aligned.
        // Otherwise, performance may suffer significantly.
        TFLITE_DCHECK_EQ(  // NOLINT
            (uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1), 0);
        const int8x8_t s1_8x8 = vld1_s8((const int8_t*)(aligned_vec + col));
        const int8x8_t s2_8x8 = vld1_s8((const int8_t*)(row_ptr + col));
        const int16x8_t prod_16x8 = vmull_s8(s1_8x8, s2_8x8);
        dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8);
        col += (kWeightsPerNeonLane >> 1);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this row.
      int32_t dotprod = AccumulateNeonLane(dotprod_32x4);
      // Postamble loop.
      // TODO(raziel): if (ABSL_PREDICT_FALSE(col < m_cols))
      for (; col < m_cols; ++col) {
        dotprod += row_ptr[col] * aligned_vec[col];
      }  // for col

      *result += dotprod * batch_scaling_factor;
    }  // for row
  }    // for batch

  if (unaligned) {
    free(aligned_row_free);
  }
  free(aligned_vec_free);
}

inline int64x2x2_t MulAdd(int32x4_t acc, int32x4_t lhs, int32x4_t rhs) {
  int64x2x2_t result;
  const int64x2_t lhs_low = vmovl_s32(vget_low_s32(lhs));
  const int64x2_t lhs_high = vmovl_s32(vget_low_s32(lhs));
  const int64_t lhs_0 = vgetq_lane_s64(lhs_low, 0);
  const int64_t lhs_1 = vgetq_lane_s64(lhs_low, 1);
  const int64_t lhs_2 = vgetq_lane_s64(lhs_high, 0);
  const int64_t lhs_3 = vgetq_lane_s64(lhs_high, 1);

  const int64x2_t rhs_low = vmovl_s32(vget_low_s32(rhs));
  const int64x2_t rhs_high = vmovl_s32(vget_low_s32(rhs));
  const int64_t rhs_0 = vgetq_lane_s64(rhs_low, 0);
  const int64_t rhs_1 = vgetq_lane_s64(rhs_low, 1);
  const int64_t rhs_2 = vgetq_lane_s64(rhs_high, 0);
  const int64_t rhs_3 = vgetq_lane_s64(rhs_high, 1);

  const int64x2_t mul_0 = {lhs_0 * rhs_0, lhs_1 * rhs_1};
  const int64x2_t mul_1 = {lhs_2 * rhs_2, lhs_3 * rhs_3};

  result.val[0] = vaddq_s64(vmovl_s32(vget_low_s32(acc)), mul_0);
  result.val[1] = vaddq_s64(vmovl_s32(vget_high_s32(acc)), mul_1);
  return result;
}

// TODO(jaesung): Merge duplicated implementations in optimized_ops.h and
// neon_tensor_utils.cc.
inline int32x4x4_t MultiplyByQuantizedMultiplier4Rows(
    int32x4x4_t input_val, int32 quantized_multiplier, int shift) {
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  int32x4_t left_shifted_one_dup = vdupq_n_s32(1 << left_shift);
  int32x4x4_t result;
  result.val[0] =
      RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                              vmulq_s32(input_val.val[0], left_shifted_one_dup),
                              quantized_multiplier),
                          right_shift);
  result.val[1] =
      RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                              vmulq_s32(input_val.val[1], left_shifted_one_dup),
                              quantized_multiplier),
                          right_shift);
  result.val[2] =
      RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                              vmulq_s32(input_val.val[2], left_shifted_one_dup),
                              quantized_multiplier),
                          right_shift);
  result.val[3] =
      RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                              vmulq_s32(input_val.val[3], left_shifted_one_dup),
                              quantized_multiplier),
                          right_shift);
  return result;
}

void NeonApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights,
                        const int32_t* bias, int32_t layer_norm_scale_a,
                        int32_t layer_norm_scale_b, int32_t variance_limit,
                        int n_batch, int n_input, int16_t* output) {
  const int32 int16_max = std::numeric_limits<int16>::max();
  const int32 int16_min = std::numeric_limits<int16>::min();
  const int32 temp = 1048576 / n_input;

  for (int i = 0; i < n_batch; ++i) {
    int64_t sum = 0;
    int64_t sum_sq = 0;

    int j = 0;
    for (; j <= n_input - 16; j += 16) {
      const int32 index = i * n_input + j;
      const int16x8_t val_s16 = vld1q_s16(input + index);
      const int32x4_t val_s32_0 = vmovl_s16(vget_low_s16(val_s16));
      const int32x4_t val_s32_1 = vmovl_s16(vget_high_s16(val_s16));

      sum += static_cast<int64_t>(AccumulateNeonLane(val_s32_0));
      sum += static_cast<int64_t>(AccumulateNeonLane(val_s32_1));

      sum_sq += AccumulateNeonLane64(vmovl_s32(vget_low_s32(val_s32_0)));
      sum_sq += AccumulateNeonLane64(vmovl_s32(vget_high_s32(val_s32_0)));
      sum_sq += AccumulateNeonLane64(vmovl_s32(vget_low_s32(val_s32_1)));
      sum_sq += AccumulateNeonLane64(vmovl_s32(vget_high_s32(val_s32_1)));
    }
    for (; j < n_input; ++j) {
      const int32 index = i * n_input + j;
      int32 val = static_cast<int32_t>(input[index]);
      sum += val;
      sum_sq += val * val;
    }

    int32_t mean =
        static_cast<int32_t>(static_cast<int64_t>(sum) * 1024 / n_input);
    // TODO(jianlijianli): Avoids overflow but only works for POT n_input.
    int64_t variance =
        sum_sq * temp - static_cast<int64_t>(mean) * static_cast<int64_t>(mean);
    int32_t variance2 = static_cast<int32>(variance / 1048576);
    if (variance2 < 1) {
      variance2 = variance_limit;
    }
    int32_t stddev_inverse_a;
    int stddev_inverse_b;
    GetInvSqrtQuantizedMultiplierExp(variance2, /*reverse_shift*/ -1,
                                     &stddev_inverse_a, &stddev_inverse_b);

    j = 0;
    const int32x4_t mean_dup = vdupq_n_s32(mean);
    for (; j <= n_input - 32; j += 32) {
      // Load 32 items at once.
      const int32 index = i * n_input + j;
      const int16x8_t val_s16_0 = vld1q_s16(input + index);
      const int16x8_t val_s16_1 = vld1q_s16(input + index + 16);

      int32x4x4_t shifted;
      shifted.val[0] = vsubq_s32(
          vshlq_n_s32(vmovl_s16(vget_low_s16(val_s16_0)), 10), mean_dup);
      shifted.val[1] = vsubq_s32(
          vshlq_n_s32(vmovl_s16(vget_high_s16(val_s16_0)), 10), mean_dup);
      shifted.val[2] = vsubq_s32(
          vshlq_n_s32(vmovl_s16(vget_low_s16(val_s16_1)), 10), mean_dup);
      shifted.val[3] = vsubq_s32(
          vshlq_n_s32(vmovl_s16(vget_high_s16(val_s16_1)), 10), mean_dup);

      int32x4x4_t rescaled = MultiplyByQuantizedMultiplier4Rows(
          shifted, stddev_inverse_a, stddev_inverse_b);

      const int32x4_t bias_0 = vld1q_s32(bias + j);
      const int32x4_t bias_1 = vld1q_s32(bias + j + 4);
      const int32x4_t bias_2 = vld1q_s32(bias + j + 8);
      const int32x4_t bias_3 = vld1q_s32(bias + j + 12);

      const int16x8_t layer_norm_weights_s16_0 =
          vld1q_s16(layer_norm_weights + j);
      const int16x8_t layer_norm_weights_s16_1 =
          vld1q_s16(layer_norm_weights + j + 8);
      const int32x4_t layer_norm_weights_s32_0 =
          vmovl_s16(vget_low_s16(layer_norm_weights_s16_0));
      const int32x4_t layer_norm_weights_s32_1 =
          vmovl_s16(vget_high_s16(layer_norm_weights_s16_0));
      const int32x4_t layer_norm_weights_s32_2 =
          vmovl_s16(vget_low_s16(layer_norm_weights_s16_1));
      const int32x4_t layer_norm_weights_s32_3 =
          vmovl_s16(vget_high_s16(layer_norm_weights_s16_1));

      int64x2x2_t val3_0 =
          MulAdd(bias_0, rescaled.val[0], layer_norm_weights_s32_0);
      int64x2x2_t val3_1 =
          MulAdd(bias_1, rescaled.val[1], layer_norm_weights_s32_1);
      int64x2x2_t val3_2 =
          MulAdd(bias_2, rescaled.val[2], layer_norm_weights_s32_2);
      int64x2x2_t val3_3 =
          MulAdd(bias_3, rescaled.val[3], layer_norm_weights_s32_3);

      int32x4x4_t val4;
      val4.val[0] = vcombine_s32(vmovn_s64(vrshrq_n_s64(val3_0.val[0], 10)),
                                 vmovn_s64(vrshrq_n_s64(val3_0.val[1], 10)));
      val4.val[1] = vcombine_s32(vmovn_s64(vrshrq_n_s64(val3_1.val[0], 10)),
                                 vmovn_s64(vrshrq_n_s64(val3_1.val[1], 10)));
      val4.val[2] = vcombine_s32(vmovn_s64(vrshrq_n_s64(val3_2.val[0], 10)),
                                 vmovn_s64(vrshrq_n_s64(val3_2.val[1], 10)));
      val4.val[3] = vcombine_s32(vmovn_s64(vrshrq_n_s64(val3_3.val[0], 10)),
                                 vmovn_s64(vrshrq_n_s64(val3_3.val[1], 10)));

      int32x4x4_t val5_s32 = MultiplyByQuantizedMultiplier4Rows(
          val4, layer_norm_scale_a, layer_norm_scale_b + 12);
      vst1_s16(output + index, vqmovn_s32(val5_s32.val[0]));
      vst1_s16(output + index + 4, vqmovn_s32(val5_s32.val[1]));
      vst1_s16(output + index + 8, vqmovn_s32(val5_s32.val[2]));
      vst1_s16(output + index + 12, vqmovn_s32(val5_s32.val[3]));
    }
    for (; j < n_input; ++j) {
      const int32 index = i * n_input + j;
      int32 val = static_cast<int32_t>(input[index]);
      int32 shifted = 1024 * val - mean;
      int32 rescaled = MultiplyByQuantizedMultiplier(shifted, stddev_inverse_a,
                                                     stddev_inverse_b);
      // TODO(jianlijianli): Saturate this.
      int64_t val3 = rescaled * layer_norm_weights[j] + bias[j];
      int32 val4 =
          static_cast<int32>((val3 > 0 ? val3 + 512 : val3 - 512) / 1024);
      int32 val5 = MultiplyByQuantizedMultiplier(val4, layer_norm_scale_a,
                                                 layer_norm_scale_b + 12);
      val5 = std::min(std::max(int16_min, val5), int16_max);
      output[index] = static_cast<int16_t>(val5);
    }
  }
}

void NeonApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input,
                      int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    int i = 0;
#ifdef GEMMLOWP_NEON
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<int16x8_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<int16x8_t, 3>;

    for (; i <= n_input - 16; i += 16) {
      const int index = batch * n_input + i;
      F3 input0 = F3::FromRaw(vld1q_s16(input + index));
      F3 input1 = F3::FromRaw(vld1q_s16(input + index + 8));
      F0 output0 = gemmlowp::logistic(input0);
      F0 output1 = gemmlowp::logistic(input1);
      vst1q_s16(output + index, output0.raw());
      vst1q_s16(output + index + 8, output1.raw());
    }
#endif  // GEMMLOWP_NEON
    using F0_Scalar = gemmlowp::FixedPoint<int16_t, 0>;
    using F3_Scalar = gemmlowp::FixedPoint<int16_t, 3>;
    for (; i < n_input; ++i) {
      const int index = batch * n_input + i;
      F3_Scalar input_f3 = F3_Scalar::FromRaw(input[index]);
      F0_Scalar output_f0 = gemmlowp::logistic(input_f3);
      output[index] = output_f0.raw();
    }
  }
}

void NeonApplyTanh3(const int16_t* input, int32_t n_batch, int32_t n_input,
                    int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    int i = 0;
#ifdef GEMMLOWP_NEON
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<int16x8_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<int16x8_t, 3>;

    for (; i <= n_input - 16; i += 16) {
      const int index = batch * n_input + i;
      F3 input0 = F3::FromRaw(vld1q_s16(input + index));
      F3 input1 = F3::FromRaw(vld1q_s16(input + index + 8));
      F0 output0 = gemmlowp::tanh(input0);
      F0 output1 = gemmlowp::tanh(input1);
      vst1q_s16(output + index, output0.raw());
      vst1q_s16(output + index + 8, output1.raw());
    }
#endif  // GEMMLOWP_NEON
    using F0_Scalar = gemmlowp::FixedPoint<int16_t, 0>;
    using F3_Scalar = gemmlowp::FixedPoint<int16_t, 3>;
    for (; i < n_input; ++i) {
      const int index = batch * n_input + i;
      F3_Scalar input_f3 = F3_Scalar::FromRaw(input[index]);
      F0_Scalar output_f0 = gemmlowp::tanh(input_f3);
      output[index] = output_f0.raw();
    }
  }
}

void NeonApplyTanh4(const int16_t* input, int32_t n_batch, int32_t n_input,
                    int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    int i = 0;
#ifdef GEMMLOWP_NEON
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<int16x8_t, 0>;
    // F4 uses 4 integer bits, range [-16, 16], the input range expected here.
    using F4 = gemmlowp::FixedPoint<int16x8_t, 4>;

    for (; i <= n_input - 16; i += 16) {
      const int index = batch * n_input + i;
      F4 input0 = F4::FromRaw(vld1q_s16(input + index));
      F4 input1 = F4::FromRaw(vld1q_s16(input + index + 8));
      F0 output0 = gemmlowp::tanh(input0);
      F0 output1 = gemmlowp::tanh(input1);
      vst1q_s16(output + index, output0.raw());
      vst1q_s16(output + index + 8, output1.raw());
    }
#endif  // GEMMLOWP_NEON
    using F0_Scalar = gemmlowp::FixedPoint<int16_t, 0>;
    using F4_Scalar = gemmlowp::FixedPoint<int16_t, 4>;
    for (; i < n_input; ++i) {
      const int index = batch * n_input + i;
      F4_Scalar input_f4 = F4_Scalar::FromRaw(input[index]);
      F0_Scalar output_f0 = gemmlowp::tanh(input_f4);
      output[index] = output_f0.raw();
    }
  }
}

void NeonCwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
                  int n_input, int shift, int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    int i = 0;
    for (; i <= n_input - 8; i += 8) {
      const int index = batch * n_input + i;
      const int16x8_t a = vld1q_s16(input_1 + index);
      const int16x8_t b = vld1q_s16(input_2 + index);
      const int32x4_t a_s32_0 = vmovl_s16(vget_low_s16(a));
      const int32x4_t a_s32_1 = vmovl_s16(vget_high_s16(a));
      const int32x4_t b_s32_0 = vmovl_s16(vget_low_s16(b));
      const int32x4_t b_s32_1 = vmovl_s16(vget_high_s16(b));

      int32x4_t x_0 = vmulq_s32(a_s32_0, b_s32_0);
      int32x4_t x_1 = vmulq_s32(a_s32_1, b_s32_1);
      x_0 = gemmlowp::RoundingDivideByPOT(x_0, shift);
      x_1 = gemmlowp::RoundingDivideByPOT(x_1, shift);

      const int16x8_t result = vcombine_s16(vmovn_s32(x_0), vmovn_s32(x_1));
      vst1q_s16(output + index, result);
    }
    for (; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const int16_t a = input_1[index];
      const int16_t b = input_2[index];
      int64_t x = a * b;
      if (x > std::numeric_limits<std::int32_t>::max()) {
        x = std::numeric_limits<std::int32_t>::max();
      }
      const int32_t value = static_cast<int32_t>(x);
      output[index] =
          static_cast<int16_t>(gemmlowp::RoundingDivideByPOT(value, shift));
    }
  }
}

void NeonCwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
                  int n_input, int shift, int8_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    int i = 0;
    for (; i <= n_input - 8; i += 8) {
      const int index = batch * n_input + i;
      const int16x8_t a = vld1q_s16(input_1 + index);
      const int16x8_t b = vld1q_s16(input_2 + index);
      const int32x4_t a_s32_0 = vmovl_s16(vget_low_s16(a));
      const int32x4_t a_s32_1 = vmovl_s16(vget_high_s16(a));
      const int32x4_t b_s32_0 = vmovl_s16(vget_low_s16(b));
      const int32x4_t b_s32_1 = vmovl_s16(vget_high_s16(b));

      int32x4_t x_0 = vmulq_s32(a_s32_0, b_s32_0);
      int32x4_t x_1 = vmulq_s32(a_s32_1, b_s32_1);
      x_0 = gemmlowp::RoundingDivideByPOT(x_0, shift);
      x_1 = gemmlowp::RoundingDivideByPOT(x_1, shift);

      const int16x8_t result = vcombine_s16(vmovn_s32(x_0), vmovn_s32(x_1));
      vst1_s8(output + index, vmovn_s16(result));
    }
    for (; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const int16_t a = input_1[index];
      const int16_t b = input_2[index];
      int64_t x = a * b;
      if (x > std::numeric_limits<std::int32_t>::max()) {
        x = std::numeric_limits<std::int32_t>::max();
      }
      const int32_t value = static_cast<int32_t>(x);
      output[index] =
          static_cast<int8_t>(gemmlowp::RoundingDivideByPOT(value, shift));
    }
  }
}

void NeonCwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch,
                  int n_input, int16_t* output) {
  const int32 int16_max = std::numeric_limits<int16>::max();
  const int32 int16_min = std::numeric_limits<int16>::min();
  for (int batch = 0; batch < n_batch; ++batch) {
    int i = 0;
    for (; i <= n_input - 8; i += 8) {
      const int index = batch * n_input + i;
      const int16x8_t a = vld1q_s16(input_1 + index);
      const int16x8_t b = vld1q_s16(input_2 + index);
      const int32x4_t a_s32_0 = vmovl_s16(vget_low_s16(a));
      const int32x4_t a_s32_1 = vmovl_s16(vget_high_s16(a));
      const int32x4_t b_s32_0 = vmovl_s16(vget_low_s16(b));
      const int32x4_t b_s32_1 = vmovl_s16(vget_high_s16(b));

      const int32x4_t sum_0 = vaddq_s32(a_s32_0, b_s32_0);
      const int32x4_t sum_1 = vaddq_s32(a_s32_1, b_s32_1);
      vst1_s16(output + index, vqmovn_s32(sum_0));
      vst1_s16(output + index + 4, vqmovn_s32(sum_1));
    }
    for (; i < n_input; ++i) {
      const int index = batch * n_input + i;
      int32_t sum = input_1[index] + input_2[index];
      const int32 sum_clamped = std::min(int16_max, std::max(int16_min, sum));
      output[index] = static_cast<int16_t>(sum_clamped);
    }
  }
}

void NeonCwiseClipping(int16_t* input, const int16_t clipping_value,
                       int32_t n_batch, int32_t n_input) {
  const int16x8_t max_dup = vdupq_n_s16(clipping_value);
  const int16x8_t min_dup = vdupq_n_s16(-clipping_value);
  for (int batch = 0; batch < n_batch; ++batch) {
    int i = 0;
    for (; i <= n_input - 16; i += 16) {
      const int index = batch * n_input + i;
      int16x8_t val_0 = vld1q_s16(input + index);
      int16x8_t val_1 = vld1q_s16(input + index + 8);
      val_0 = vminq_s16(val_0, max_dup);
      val_1 = vminq_s16(val_1, max_dup);
      val_0 = vmaxq_s16(val_0, min_dup);
      val_1 = vmaxq_s16(val_1, min_dup);
      vst1q_s16(input + index, val_0);
      vst1q_s16(input + index + 8, val_1);
    }
    for (; i < n_input; ++i) {
      const int index = batch * n_input + i;
      if (input[index] > clipping_value) {
        input[index] = clipping_value;
      }
      if (input[index] < -clipping_value) {
        input[index] = -clipping_value;
      }
    }
  }
}

void NeonCwiseClipping(int8_t* input, const int8_t clipping_value,
                       int32_t n_batch, int32_t n_input) {
  const int8x16_t max_dup = vdupq_n_s8(clipping_value);
  const int8x16_t min_dup = vdupq_n_s8(-clipping_value);
  for (int batch = 0; batch < n_batch; ++batch) {
    int i = 0;
    for (; i <= n_input - 32; i += 32) {
      const int index = batch * n_input + i;
      int8x16_t val_0 = vld1q_s8(input + index);
      int8x16_t val_1 = vld1q_s8(input + index + 16);
      val_0 = vminq_s8(val_0, max_dup);
      val_1 = vminq_s8(val_1, max_dup);
      val_0 = vmaxq_s8(val_0, min_dup);
      val_1 = vmaxq_s8(val_1, min_dup);
      vst1q_s8(input + index, val_0);
      vst1q_s8(input + index + 16, val_1);
    }
    for (; i < n_input; ++i) {
      const int index = batch * n_input + i;
      if (input[index] > clipping_value) {
        input[index] = clipping_value;
      }
      if (input[index] < -clipping_value) {
        input[index] = -clipping_value;
      }
    }
  }
}

void NeonSparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result, int result_stride) {
  const int kBlockSize = 16;
  const int kNeonVectorsPerBlock = 4;
  TFLITE_DCHECK_EQ(  // NOLINT
      m_cols % kBlockSize, 0);

  float* result_in_batch = result;
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    const uint8_t* ledger_ptr = ledger;
    for (int r = 0; r < m_rows; r++) {
      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {
        float32x4_t acc_32x4 = vmovq_n_f32(0.0);
        const float* vector_in_batch = vector + b * m_cols;

        for (int i = 0; i < num_nonzero_blocks; i++) {
          const int block_start_index = *ledger_ptr++ * kBlockSize;
          const float* vector_block_in_batch_ptr =
              vector_in_batch + block_start_index;

          for (int c = 0; c < kNeonVectorsPerBlock; c++) {
            // Load 4 float values from the vector and matrix row.
            float32x4_t vector_f32x4 = vld1q_f32(vector_block_in_batch_ptr +
                                                 c * kFloatValuesPerNeonVector);
            float32x4_t matrix_f32x4 =
                vld1q_f32(matrix_ptr + c * kFloatValuesPerNeonVector);
            // Multiply the vector and matrix row and add to accumulator.
            acc_32x4 = vmlaq_f32(acc_32x4, matrix_f32x4, vector_f32x4);
          }
          matrix_ptr += kBlockSize;
        }
        *result_in_batch += AccumulateNeonLane(acc_32x4);
      }
      result_in_batch += result_stride;
    }
  }
}

void NeonSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    int result_stride) {
#ifdef __aarch64__
  if (HasSdotInstruction() && m_cols % 16 == 0) {
    DotprodSparseMatrixBatchVectorMultiplyAccumulate(
        matrix, ledger, m_rows, m_cols, vectors, scaling_factors, n_batch,
        result, result_stride);
    return;
  }
#endif  // __aarch64__

  const int kWeightsPerUint32 = 4;
  const int kWeightsPerNeonLane = 16;
  const int kBlockSize = kWeightsPerNeonLane;
  TFLITE_DCHECK_EQ(  // NOLINT
      m_cols % kBlockSize, 0);
  void* aligned_vec_free = nullptr;
  int8_t* aligned_vec =
      (int8_t*)aligned_alloc(kWeightsPerUint32, m_cols,  // NOLINT
                             &aligned_vec_free);

  for (int batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    // Copy the vector data to an aligned vector.
    memcpy(aligned_vec, vectors + batch * m_cols, sizeof(int8) * m_cols);

    const uint8_t* ledger_ptr = ledger;
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row, result += result_stride) {
      // Initialize the dot product sum for the row to 0.
      int32x4_t dotprod_32x4 = vmovq_n_s32(0);
      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {
        // Prefetch the row to cache.
        __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                           3 /* temporal locality */);
        for (int i = 0; i < num_nonzero_blocks; i++) {
          const int col_index = *ledger_ptr++ * kBlockSize;
          // Load 16 8-bit values from the row and vector, each, to operate on.
          // Here the assumption is that each buffer is 4-byte aligned.
          // Otherwise, performance may suffer significantly.
          TFLITE_DCHECK_EQ(  // NOLINT
              (uintptr_t)(&row_ptr) & (kWeightsPerUint32 - 1), 0);
          const int8x16_t s1_8x16 =
              vld1q_s8((const int8_t*)(aligned_vec + col_index));
          const int8x16_t s2_8x16 = vld1q_s8((const int8_t*)(row_ptr));
          // Multiply the low bits (i.e. the lower 8 8bit numbers in the
          // registers).
          int16x8_t prod_16x8 =
              vmull_s8(vget_low_s8(s1_8x16), vget_low_s8(s2_8x16));
          // Multiply the high bits (i.e. the lower 8 8bit numbers in the
          // registers), and accumulate with the result of the low bits product.
          // The assumption here is that overflow will not happen as we quantize
          // our values to be in the range [-127, 127]. As such the sum of the 2
          // products is always strictly smaller than 15-bits (32767 in absolute
          // value).
          prod_16x8 =
              vmlal_s8(prod_16x8, vget_high_s8(s1_8x16), vget_high_s8(s2_8x16));

          dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8);
          row_ptr += kBlockSize;
        }
        // Add the 4 intermediate sum values to get the final dot-prod value for
        // this row.
        int32_t dotprod = AccumulateNeonLane(dotprod_32x4);
        *result += dotprod * batch_scaling_factor;
      }
    }  // for row
  }    // for batch
  free(aligned_vec_free);
}

void NeonVectorVectorCwiseProduct(const float* vector1, const float* vector2,
                                  int v_size, float* result) {
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownToFloatVectors(v_size);
  int v = 0;
  for (; v < postamble_start; v += kFloatValuesPerNeonVector) {
    // Load 4 float values from vector1 and vector2.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    // Vector multiply 4 float
    float32x4_t mul_32x4 = vmulq_f32(v1_f32x4, v2_f32x4);
    // Save to result array.
    vst1q_f32(&result[v], mul_32x4);
  }
  for (; v < v_size; v++) {
    result[v] = vector1[v] * vector2[v];
  }
}

void NeonVectorVectorCwiseProductAccumulate(const float* vector1,
                                            const float* vector2, int v_size,
                                            float* result) {
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownToFloatVectors(v_size);
  int v = 0;
  for (; v < postamble_start; v += kFloatValuesPerNeonVector) {
    // Load 4 float values from vector1 and vector2 and accumulator.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    float32x4_t acc_32x4 = vld1q_f32(result + v);
    // Vector multiply-accumulate 4 float
    acc_32x4 = vmlaq_f32(acc_32x4, v1_f32x4, v2_f32x4);
    // Save to result array.
    vst1q_f32(&result[v], acc_32x4);
  }
  for (; v < v_size; v++) {
    result[v] += vector1[v] * vector2[v];
  }
}

void NeonVectorBatchVectorCwiseProduct(const float* vector, int v_size,
                                       const float* batch_vector, int n_batch,
                                       float* result) {
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownToFloatVectors(v_size);

  for (int b = 0; b < n_batch; b++) {
    int v = 0;
    for (; v < postamble_start; v += kFloatValuesPerNeonVector) {
      // Load from memory to vectors.
      float32x4_t batch_vector_f32x4 = vld1q_f32(batch_vector + v);
      float32x4_t vector_f32x4 = vld1q_f32(vector + v);
      // Multiply.
      float32x4_t result_f32x4 = vmulq_f32(batch_vector_f32x4, vector_f32x4);
      // Store.
      vst1q_f32(result + v, result_f32x4);
    }
    // Postamble loop
    for (; v < v_size; v++) {
      result[v] = vector[v] * batch_vector[v];
    }
    // Update the pointers.
    result += v_size;
    batch_vector += v_size;
  }
}

void NeonVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                 int v_size,
                                                 const float* batch_vector,
                                                 int n_batch, float* result) {
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownToFloatVectors(v_size);

  float* result_ptr = result;
  const float* batch_vector_ptr = batch_vector;
  for (int b = 0; b < n_batch; b++) {
    int v = 0;
    for (; v < postamble_start; v += kFloatValuesPerNeonVector) {
      // Load from memory to vectors.
      float32x4_t result_f32x4 = vld1q_f32(result_ptr + v);
      float32x4_t batch_vector_f32x4 = vld1q_f32(batch_vector_ptr + v);
      float32x4_t vector_f32x4 = vld1q_f32(vector + v);
      // Multiply-accumulate.
      result_f32x4 = vmlaq_f32(result_f32x4, batch_vector_f32x4, vector_f32x4);
      // Store.
      vst1q_f32(result_ptr + v, result_f32x4);
    }
    // Postamble loop
    for (; v < v_size; v++) {
      result_ptr[v] += vector[v] * batch_vector_ptr[v];
    }
    // Update the pointers.
    result_ptr += v_size;
    batch_vector_ptr += v_size;
  }
}

void NeonSub1Vector(const float* vector, int v_size, float* result) {
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownToFloatVectors(v_size);

  float32x4_t one_f32x4 = vmovq_n_f32(1.0);
  int v = 0;
  for (; v < postamble_start; v += kFloatValuesPerNeonVector) {
    // Load 4 float values from the current pointers of the input column and
    // subtract from 1.
    float32x4_t v_f32x4 = vld1q_f32(vector + v);
    float32x4_t result_f32x4 = vsubq_f32(one_f32x4, v_f32x4);
    // Save to output.
    vst1q_f32(result + v, result_f32x4);
  }
  for (; v < v_size; v++) {
    result[v] = 1.0f - vector[v];
  }
}

bool NeonIsZeroVector(const float* vector, int v_size) {
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownToFloatVectors(v_size);

  const float32x4_t zero_x4_float = vmovq_n_f32(0.0f);
  int v = 0;
  for (; v < postamble_start; v += kFloatValuesPerNeonVector) {
    const float32x4_t i_x4_float = vld1q_f32(vector + v);
    uint32x4_t cmp_result = vceqq_f32(i_x4_float, zero_x4_float);
    if (vgetq_lane_u32(cmp_result, 0) == 0) return false;
    if (vgetq_lane_u32(cmp_result, 1) == 0) return false;
    if (vgetq_lane_u32(cmp_result, 2) == 0) return false;
    if (vgetq_lane_u32(cmp_result, 3) == 0) return false;
  }
  // Postamble loop
  for (; v < v_size; ++v) {
    if (vector[v] != 0.0) return false;
  }
  return true;
}

void NeonClipVector(const float* vector, int v_size, float abs_limit,
                    float* result) {
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownToFloatVectors(v_size);

  // Replicate abs_limit and -abs_limit in two vectors.
  const float32x4_t abs_limit_f32x4 = vmovq_n_f32(abs_limit);
  const float32x4_t neg_abs_limit_f32x4 = vmovq_n_f32(-abs_limit);

  int v = 0;
  for (; v < postamble_start; v += kFloatValuesPerNeonVector) {
    // Load from memory to vector.
    float32x4_t v_f32x4 = vld1q_f32(vector + v);
    // Clip between abs_limit and -abs_limit.
    float32x4_t result_f32x4 = vminq_f32(abs_limit_f32x4, v_f32x4);
    result_f32x4 = vmaxq_f32(neg_abs_limit_f32x4, result_f32x4);
    // Save to output.
    vst1q_f32(result + v, result_f32x4);
  }
  // Postamble loop.
  for (; v < v_size; v++) {
    result[v] = (abs_limit < vector[v]) ? abs_limit : vector[v];
    result[v] = (-abs_limit > result[v]) ? -abs_limit : result[v];
  }
}

void NeonVectorScalarMultiply(const int8_t* vector, const int v_size,
                              const float scale, float* result) {
  // Here the assumption is that each buffer is 4-byte aligned.
  const int kWeightsPerUint32 = 4;
  TFLITE_CHECK_EQ((intptr_t)(&vector[0]) & (kWeightsPerUint32 - 1), 0);
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int kWeightsPerNeonLane = 16;
  const int postamble_start = v_size - (v_size & (kWeightsPerNeonLane - 1));

  // Create a vector of 4 floats with the scale value.
  const float32x4_t scale_f32x4 = vdupq_n_f32(scale);
  int v = 0;
  for (; v < postamble_start; v += kWeightsPerNeonLane) {
    // Load int8 values, sixteen at a time.
    const int8x16_t v_i8x16 = vld1q_s8(vector + v);
    // Split it into two components of size eight.
    const int8x8_t v0_i8x8 = vget_low_s8(v_i8x16);
    const int8x8_t v1_i8x8 = vget_high_s8(v_i8x16);
    // Convert both components to int16 first.
    const int16x8_t v0_i16x8 = vmovl_s8(v0_i8x8);
    const int16x8_t v1_i16x8 = vmovl_s8(v1_i8x8);
    // Split each of them into two components each.
    const int16x4_t v0_i16x4 = vget_low_s16(v0_i16x8);
    const int16x4_t v1_i16x4 = vget_high_s16(v0_i16x8);
    const int16x4_t v2_i16x4 = vget_low_s16(v1_i16x8);
    const int16x4_t v3_i16x4 = vget_high_s16(v1_i16x8);
    // Convert these to int32 and then to float.
    float32x4_t v0_f32x4 = vcvtq_f32_s32(vmovl_s16(v0_i16x4));
    float32x4_t v1_f32x4 = vcvtq_f32_s32(vmovl_s16(v1_i16x4));
    float32x4_t v2_f32x4 = vcvtq_f32_s32(vmovl_s16(v2_i16x4));
    float32x4_t v3_f32x4 = vcvtq_f32_s32(vmovl_s16(v3_i16x4));
    // Vector multiply four floats at a time.
    v0_f32x4 = vmulq_f32(v0_f32x4, scale_f32x4);
    v1_f32x4 = vmulq_f32(v1_f32x4, scale_f32x4);
    v2_f32x4 = vmulq_f32(v2_f32x4, scale_f32x4);
    v3_f32x4 = vmulq_f32(v3_f32x4, scale_f32x4);
    // Store the results.
    vst1q_f32(result + v, v0_f32x4);
    vst1q_f32(result + v + 4, v1_f32x4);
    vst1q_f32(result + v + 8, v2_f32x4);
    vst1q_f32(result + v + 12, v3_f32x4);
  }

  if (v_size - postamble_start >= (kWeightsPerNeonLane >> 1)) {
    // Load eight int8 values, if there is at least eight remaining.
    const int8x8_t v_i8x8 = vld1_s8(vector + v);
    // Convert them to int16 first.
    const int16x8_t v_i16x8 = vmovl_s8(v_i8x8);
    // Split it into two components.
    const int16x4_t v0_i16x4 = vget_low_s16(v_i16x8);
    const int16x4_t v1_i16x4 = vget_high_s16(v_i16x8);
    // Convert the components two floats.
    float32x4_t v0_f32x4 = vcvtq_f32_s32(vmovl_s16(v0_i16x4));
    float32x4_t v1_f32x4 = vcvtq_f32_s32(vmovl_s16(v1_i16x4));
    // Vector multiply four floats at a time.
    v0_f32x4 = vmulq_f32(v0_f32x4, scale_f32x4);
    v1_f32x4 = vmulq_f32(v1_f32x4, scale_f32x4);
    // Store the results.
    vst1q_f32(result + v, v0_f32x4);
    vst1q_f32(result + v + 4, v1_f32x4);
    v += (kWeightsPerNeonLane >> 1);
  }

  // Postamble loop.
  for (; v < v_size; v++) {
    result[v] = scale * vector[v];
  }
}

// TODO(renjieliu): Avoid duplicating the logic.
// Also consider changing the rounding stragey from "ties to away" to
// "ties to even" since vcvtnq_s32_f32 is generally more available.
inline int32x4_t RoundToNearest(const float32x4_t input) {
#if defined(_ACAT_ARM64)
  return vcvtaq_s32_f32(input);
#else
  static const float32x4_t zero_val_dup = vdupq_n_f32(0.0f);
  static const float32x4_t point5_val_dup = vdupq_n_f32(0.5f);

  const int32x4_t mask = vreinterpretq_s32_u32(vcltq_f32(input, zero_val_dup));
  const float32x4_t casted_mask = vcvtq_f32_s32(mask);
  const float32x4_t round = vaddq_f32(casted_mask, point5_val_dup);
  return vcvtq_s32_f32(vaddq_f32(input, round));
#endif
}

void NeonSymmetricQuantizeFloats(const float* values, const int size,
                                 int8_t* quantized_values, float* min,
                                 float* max, float* scaling_factor) {
  // TODO(raziel): vectorize min/max calculation.
  auto minmax = std::minmax_element(values, values + size);
  *min = *minmax.first;
  *max = *minmax.second;
  const int kScale = 127;
  const float range = std::max(std::abs(*min), std::abs(*max));
  if (range == 0) {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;

  const int postamble_start = size & ~(2 * kFloatValuesPerNeonVector - 1);

  // Vectorized constants.
  const float32x4_t q_factor_f32x4 = vmovq_n_f32(scaling_factor_inv);
  const int32x4_t scale_i32x4 = vmovq_n_s32(kScale);
  const int32x4_t neg_scale_i32x4 = vmovq_n_s32(-kScale);

  int i = 0;
  for (; i < postamble_start; i += 2 * kFloatValuesPerNeonVector) {
    // Implements the vectorized version of the following:
    // const int32 quantized_value = static_cast<int32>(
    //    std::round(*scaling_factor * values[i]));
    float32x4_t value0_f32x4 = vld1q_f32(&values[i]);
    float32x4_t value1_f32x4 =
        vld1q_f32(&values[i + kFloatValuesPerNeonVector]);
    float32x4_t mul0_f32x4 = vmulq_f32(value0_f32x4, q_factor_f32x4);
    float32x4_t mul1_f32x4 = vmulq_f32(value1_f32x4, q_factor_f32x4);

    const int32x4_t f2i0_i32x4 = RoundToNearest(mul0_f32x4);
    const int32x4_t f2i1_i32x4 = RoundToNearest(mul1_f32x4);

    // Implements the vectorized version of the folowing block:
    //  quantized_values[i] = std::min(kScale, std::max(-kScale,
    //  quantized_value));
    int32x4_t max0_i32x4 = vmaxq_s32(f2i0_i32x4, neg_scale_i32x4);
    int32x4_t max1_i32x4 = vmaxq_s32(f2i1_i32x4, neg_scale_i32x4);
    int32x4_t min0_i32x4 = vminq_s32(max0_i32x4, scale_i32x4);
    int32x4_t min1_i32x4 = vminq_s32(max1_i32x4, scale_i32x4);

    int16x4_t min0_16x4 = vmovn_s32(min0_i32x4);
    int16x4_t min1_16x4 = vmovn_s32(min1_i32x4);

    int16x8_t min_16x8 = vcombine_s16(min0_16x4, min1_16x4);
    int8x8_t min_s8x8 = vqmovn_s16(min_16x8);
    vst1_s8(&quantized_values[i], min_s8x8);
  }

  for (; i < size; ++i) {
    const int32 quantized_value =
        static_cast<int32>(TfLiteRound(scaling_factor_inv * values[i]));
    quantized_values[i] = std::min(kScale, std::max(-kScale, quantized_value));
  }
}

float NeonVectorVectorDotProduct(const float* vector1, const float* vector2,
                                 int v_size) {
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownToFloatVectors(v_size);
  float32x4_t acc_32x4 = vmovq_n_f32(0.0);
  int v = 0;
  for (; v < postamble_start; v += kFloatValuesPerNeonVector) {
    // Load 4 float values from vector1 and vector2 and accumulator.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    // Vector multiply-accumulate 4 float
    acc_32x4 = vmlaq_f32(acc_32x4, v1_f32x4, v2_f32x4);
  }
  float result = AccumulateNeonLane(acc_32x4);
  // Postamble loop.
  for (; v < v_size; v++) {
    result += vector1[v] * vector2[v];
  }
  return result;
}

void NeonBatchVectorBatchVectorDotProduct(const float* vector1,
                                          const float* vector2, int v_size,
                                          int n_batch, float* result,
                                          int result_stride) {
  float* result_ptr = result;
  const float* vector1_ptr = vector1;
  const float* vector2_ptr = vector2;
  for (int b = 0; b < n_batch; b++) {
    *result_ptr = NeonVectorVectorDotProduct(vector1_ptr, vector2_ptr, v_size);
    vector1_ptr += v_size;
    vector2_ptr += v_size;
    result_ptr += result_stride;
  }
}

void NeonReductionSumVector(const float* input_vector, float* output_vector,
                            int output_size, int reduction_size) {
  const float* input_vector_ptr = input_vector;
  for (int o = 0; o < output_size; o++) {
    // If v_size is not divisible by the vector size, then we need to process
    // the final few elements sequentially. postamble_start shows the start
    // index where this should happen.
    const int postamble_start = RoundDownToFloatVectors(reduction_size);
    float32x4_t sum_f32x4 = vmovq_n_f32(0.0);
    int r = 0;
    for (; r < postamble_start; r += kFloatValuesPerNeonVector) {
      float32x4_t v1_f32x4 = vld1q_f32(input_vector_ptr + r);
      sum_f32x4 = vaddq_f32(sum_f32x4, v1_f32x4);
    }
    output_vector[o] += AccumulateNeonLane(sum_f32x4);
    input_vector_ptr += postamble_start;

    // Postamble loop.
    for (; r < reduction_size; r++) {
      output_vector[o] += *input_vector_ptr++;
    }
  }
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // USE_NEON
