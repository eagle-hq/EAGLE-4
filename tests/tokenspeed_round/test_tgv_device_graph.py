# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Small BF16 projections captured in a device-launched draft graph."""


import pytest
import torch

pytest.importorskip("tvm_ffi", reason="TokenSpeed round kernels require apache-tvm-ffi")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

pytest.importorskip("tokenspeed_kernel", reason="requires the optional TokenSpeed kernels")

from tokenspeed_kernel.ops.gemm import flashinfer as gemm_flashinfer

if not hasattr(gemm_flashinfer, "has_flashinfer_cute_dsl_bf16"):
    pytest.skip("engine has no optional TGV BF16 backend", allow_module_level=True)

from tokenspeed_kernel.ops.gemm.flashinfer import (
    flashinfer_bf16_gemm,
    has_flashinfer_cute_dsl_bf16,
)

from eagle_harness.tokenspeed.tree_round.chain import DeviceChain

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_flashinfer_cute_dsl_bf16(),
    reason="TGV needs CuTe DSL on SM100/SM103",
)


@pytest.mark.parametrize("m,k", [(1, 1024), (7, 2048), (16, 5120)])
def test_small_bf16_gemm_replays_in_a_device_launched_child(m, k):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(53)
    x = torch.randn(m, k, device=device, generator=gen).bfloat16()
    weight = (torch.randn(5120, k, device=device, generator=gen) * 0.05).bfloat16()
    out = torch.empty(m, 5120, dtype=x.dtype, device=device)
    observed = torch.empty_like(out)
    go = torch.ones(3, dtype=torch.int32, device=device)
    sampled = torch.zeros(1, dtype=torch.bool, device=device)
    chain = DeviceChain(1, 2, device)
    stream = torch.cuda.Stream()
    flashinfer_bf16_gemm(x, weight, out)
    torch.cuda.synchronize()
    first = torch.cuda.CUDAGraph()
    child = torch.cuda.CUDAGraph(keep_graph=True)
    final = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(first, stream=stream):
            chain.end_depth(1, 1, go, sampled)
        with torch.cuda.graph(child, stream=stream):
            flashinfer_bf16_gemm(x, weight, out)
            chain.end_depth(1, 2, go, sampled)
        with torch.cuda.graph(final, stream=stream):
            observed.copy_(out)
        chain.instantiate(1, [child], [final], stream)
        stream.synchronize()
        for _ in range(4):
            x.copy_(torch.randn(m, k, device=device, generator=gen).bfloat16())
            expected = (x.double() @ weight.double().T).bfloat16()
            out.fill_(float("nan"))
            observed.fill_(float("nan"))
            first.replay()
            torch.testing.assert_close(observed, expected, rtol=1e-2, atol=2e-3)
    finally:
        torch.cuda.synchronize()
        chain.release()
