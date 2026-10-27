/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Modified by Eagle AI Corp.

#pragma once
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <dlpack/dlpack.h>
using tvm::ffi::TensorView;
constexpr DLDataType dl_int32 = DLDataType{kDLInt, 32, 1};
constexpr DLDataType dl_int64 = DLDataType{kDLInt, 64, 1};
constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_float64 = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType dl_bool = DLDataType{kDLBool, 8, 1};

inline cudaStream_t get_stream(DLDevice device) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}

inline int64_t get_element_size(TensorView x) { return (x.dtype().bits * x.dtype().lanes) / 8; }

inline void check_tensor(const TensorView& t, DLDataType dtype, DLDevice device) {
  TVM_FFI_ICHECK(t.dtype() == dtype && t.device().device_type == kDLCUDA &&
                 t.device().device_id == device.device_id)
      << "EAGLE-Harness: tensor dtype/device mismatch";
}

inline void check_scalar(const TensorView& t, DLDataType dtype, DLDevice device) {
  check_tensor(t, dtype, device);
  TVM_FFI_ICHECK(t.numel() == 1 && t.IsContiguous()) << "EAGLE-Harness: expected one device scalar";
}
