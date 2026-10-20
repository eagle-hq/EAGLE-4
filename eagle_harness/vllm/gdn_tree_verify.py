# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Linear-attention tree verify for vLLM hybrid targets: GDN (Qwen3.5 / 3.8) and Mamba2 (NemotronH).

Plugs into vLLM's ``LinearAttnVerifyExtension``: the Qwen GDN layer hands it the speculative-decode rows of every
forward (``GDNSpecRows``) and the Mamba2 layer its decode rows (``Mamba2SpecRows``) in place of their own conv and
recurrence kernels. vLLM verifies a chain with one state row per verify row: row ``i`` of request ``b`` stores its
recurrent state in the state slot ``slots[b, i]`` (the request's block table), the convolution window of the request
lives in slot ``slots[b, 0]`` as the raw inputs of the last tokens, and the next step resumes from row / window offset
``num_accepted - 1``.

A tree keeps that layout and runs the shared tree kernels on it:

- the conv (:func:`~eagle_harness.kernels.gdn_tree_conv.gdn_tree_conv1d_update`, a width-4 causal conv that serves
  GDN's qkv and Mamba2's xBC alike) gathers each row's three earlier conv inputs through the ancestor table of
  :class:`~eagle_harness.vllm.verify_layout.VerifyStage` (a row sees its ancestors, not the rows before it) and rolls
  the window as the chain kernel does: ``[old[1:], x_0, .., x_{n-1}]``;
- the recurrence walks each request's rows with vLLM's slot table: row ``i`` stores its state in ``slots[b, i]`` and
  starts from its parent's, row 0 from ``slots[b, num_accepted - 1]``. GDN runs the shared tree walk
  (:func:`eagle_harness.kernels.gdn_tree_recurrence.gdn_tree_recurrent_update`, and its Gluon twin where it applies),
  Mamba2 the tree selective state update (:func:`eagle_harness.kernels.mamba2_tree.mamba2_tree_state_update`).

The accepted rows are not a prefix of the verify rows, so :class:`LinearStateCommit` (run from
``post_accept``) moves them into chain order: the window columns of the accepted path next to each other,
and the last accepted row's recurrent state into row ``accepted - 1``, where vLLM's next step resumes from.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import triton
import triton.language as tl

from eagle_harness.vllm.adapter import verify_extension_module
from eagle_harness.kernels.gdn_tree_conv import gdn_tree_conv1d_update
from eagle_harness.kernels.gdn_tree_recurrence import gdn_tree_recurrent_update
from eagle_harness.kernels.mamba2_tree import mamba2_tree_state_update
from eagle_harness.vllm.verify_layout import VerifyStage, get_stage

if TYPE_CHECKING:  # vLLM builds without the Mamba2 hook lack it; the speculator refuses Mamba2 targets there
    from vllm.model_executor.layers.mamba.verify_extension import Mamba2SpecRows

_verify_extension = verify_extension_module()
GDNSpecRows = _verify_extension.GDNSpecRows
LinearAttnVerifyExtension = _verify_extension.LinearAttnVerifyExtension

# A padded request's state slot; the chain kernels skip it too.
NULL_BLOCK_ID = tl.constexpr(0)


class TreeVerify(LinearAttnVerifyExtension):
    """Runs a GDN or Mamba2 layer's spec-decode rows through the tree kernels with the staged layout.

    Trees when the stage verifies trees; otherwise the stage holds chain layouts (a mixed batch), which the same
    kernels compute as vLLM's chain kernels do.
    """

    def __init__(self, vllm_config: Any):
        self.vllm_config = vllm_config

    def forward_spec_rows(self, rows: GDNSpecRows) -> None:
        stage = _staged()
        tokens, dim = rows.mixed_qkv.shape
        conv = torch.empty(tokens, dim, dtype=rows.mixed_qkv.dtype, device=rows.mixed_qkv.device)
        gdn_tree_conv1d_update(
            rows.mixed_qkv.t()[None],
            rows.conv_state,
            rows.conv_weight,
            rows.conv_bias,
            rows.activation,
            conv_state_indices=rows.state_indices[:, 0],
            intermediate_conv_window=None,
            intermediate_state_indices=None,
            tree_ancestors=stage.ancestors[: rows.query_start_loc.shape[0] - 1],
            pad_slot_id=NULL_BLOCK_ID.value,
            out=conv.t()[None],
            num_accepted_tokens=rows.num_accepted_tokens,
            query_start_loc=rows.query_start_loc,
        )
        key_dim = rows.num_k_heads * rows.head_k_dim
        q, k, v = conv.split([key_dim, key_dim, rows.num_v_heads * rows.head_v_dim], dim=-1)
        published = stage.state_slots.get(rows.layer_name)  # one layer per KV cache group publishes the group's slots
        gdn_tree_recurrent_update(
            rows.A_log,
            rows.dt_bias,
            q.view(1, tokens, -1, rows.head_k_dim),
            k.view(1, tokens, -1, rows.head_k_dim),
            v.view(1, tokens, -1, rows.head_v_dim),
            rows.a,
            rows.b,
            initial_state_source=rows.ssm_state,
            initial_state_indices=rows.state_indices,
            num_accepted_tokens=rows.num_accepted_tokens,
            cu_seqlens=rows.query_start_loc,
            intermediate_states_buffer=rows.ssm_state,
            intermediate_state_indices=rows.state_indices,
            retrieve_parent_token=stage.parents,
            pad_slot_id=NULL_BLOCK_ID.value,
            out=rows.out.view(1, tokens, -1, rows.head_v_dim),
            publish=None if published is None else published[: rows.state_indices.shape[0]],
            gluon=True,
        )

    def forward_mamba2_spec_rows(self, rows: Mamba2SpecRows) -> None:
        stage = _staged()
        tokens, dim = rows.hidden_states_B_C.shape
        reqs = rows.state_indices.shape[0]
        conv = torch.empty(tokens, dim, dtype=rows.hidden_states_B_C.dtype, device=rows.hidden_states_B_C.device)
        gdn_tree_conv1d_update(
            rows.hidden_states_B_C.t()[None],
            rows.conv_state,
            rows.conv_weight,
            rows.conv_bias,
            rows.activation,
            conv_state_indices=rows.state_indices[:, 0],
            intermediate_conv_window=None,
            intermediate_state_indices=None,
            tree_ancestors=stage.ancestors[: rows.query_start_loc.shape[0] - 1],
            pad_slot_id=NULL_BLOCK_ID.value,
            out=conv.t()[None],
            num_accepted_tokens=rows.num_accepted_tokens,
            query_start_loc=rows.query_start_loc,
        )
        heads, head_dim, groups, dstate = rows.num_heads, rows.head_dim, rows.n_groups, rows.ssm_state_size
        x, B, C = conv.split([heads * head_dim, groups * dstate, groups * dstate], dim=-1)
        mamba2_tree_state_update(
            rows.ssm_state,
            x.view(tokens, heads, head_dim),
            rows.dt,
            rows.A,
            B.view(tokens, groups, dstate),
            C.view(tokens, groups, dstate),
            rows.D,
            rows.dt_bias,
            rows.out.view(tokens, heads, head_dim),
            state_indices=rows.state_indices,
            parents=stage.parents,
            query_start_loc=rows.query_start_loc,
            num_accepted_tokens=rows.num_accepted_tokens,
            pad_slot_id=NULL_BLOCK_ID.value,
        )
        published = stage.state_slots.get(rows.layer_name)  # one layer per KV cache group publishes the group's slots
        if published is not None:
            published[:reqs].copy_(rows.state_indices)


def check_tree_verifiable(layers: Sequence[Any]) -> None:
    """Refuse a target with linear-attention layers that do not hand their rows to this extension: their in-tree
    kernels would verify every tree as a chain."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    handled: tuple[type, ...] = (QwenGatedDeltaNetAttention,)
    if "forward_mamba2_spec_rows" in vars(LinearAttnVerifyExtension):
        handled += (MambaMixer2,)
    unhandled = sorted({type(layer).__name__ for layer in layers if not isinstance(layer, handled)})
    if unhandled:
        raise RuntimeError(
            f"eagle_harness: tree verify cannot run on {', '.join(unhandled)} layers with this vLLM (only "
            f"{', '.join(t.__name__ for t in handled)} hand their rows to the verify extension); Mamba2 needs vLLM's "
            "Mamba2 verify hook (LinearAttnVerifyExtension.forward_mamba2_spec_rows)"
        )


def _staged() -> VerifyStage:
    stage = get_stage()
    if stage is None:
        raise RuntimeError("eagle_harness: a linear-attention verify ran before the speculator staged its layout")
    return stage


def create_extension(vllm_config: Any) -> TreeVerify | None:
    """``SpeculativeMethodSpec.linear_attn_verify_extension`` factory."""
    from eagle_harness.vllm.config import parse_method_config

    if parse_method_config(vllm_config.speculative_config.method_config).node_top_k == 1:
        return None  # a chain is vLLM's own verify
    return TreeVerify(vllm_config)


@triton.jit
def _compact_window_kernel(
    conv_ptrs,  # [L] uint64: base address of each layer's window cache
    slots_ptrs,  # [L] uint64: the layer's group's slots, [reqs, S] int32 (column 0: the window slot)
    slots_strides,  # [L] int64
    idx_mapping_ptr,
    path_ptr,  # [state slots, W] int32
    stride_path,
    path_len_ptr,
    stride_slot,
    stride_dim,
    stride_tok,
    dim,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    b, layer, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    state = tl.load(idx_mapping_ptr + b)
    slots_ptr = tl.load(slots_ptrs + layer).to(tl.pointer_type(tl.int32))
    slot = tl.load(slots_ptr + b * tl.load(slots_strides + layer)).to(tl.int64)
    if slot <= NULL_BLOCK_ID:
        return
    feats = block * BLOCK + tl.arange(0, BLOCK)
    mask = feats < dim
    base = tl.load(conv_ptrs + layer).to(tl.pointer_type(DTYPE)) + slot * stride_slot + feats * stride_dim
    # Path row j moves from column 2 + path[j] to column 2 + j. Ascending, so no source is overwritten first.
    for j in range(1, tl.load(path_len_ptr + state)):
        src = 2 + tl.load(path_ptr + state * stride_path + j)
        if src != 2 + j:
            tl.store(base + (2 + j) * stride_tok, tl.load(base + src * stride_tok, mask=mask), mask=mask)
            tl.debug_barrier()


@triton.jit
def _move_state_kernel(
    ssm_ptrs,  # [L] uint64
    slots_ptrs,  # [L] uint64: the layer's group's slots, [reqs, S] int32
    slots_strides,  # [L] int64
    idx_mapping_ptr,
    path_ptr,
    stride_path,
    path_len_ptr,
    stride_slot,
    numel,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    b, layer, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    state = tl.load(idx_mapping_ptr + b)
    length = tl.load(path_len_ptr + state)
    last = tl.load(path_ptr + state * stride_path + length - 1)
    slots_ptr = tl.load(slots_ptrs + layer).to(tl.pointer_type(tl.int32)) + b * tl.load(slots_strides + layer)
    src_slot = tl.load(slots_ptr + last).to(tl.int64)
    dst_slot = tl.load(slots_ptr + length - 1).to(tl.int64)
    if (last != length - 1) & (src_slot > NULL_BLOCK_ID) & (dst_slot > NULL_BLOCK_ID):
        base = tl.load(ssm_ptrs + layer).to(tl.pointer_type(DTYPE))
        offs = block * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        tl.store(
            base + dst_slot * stride_slot + offs, tl.load(base + src_slot * stride_slot + offs, mask=mask), mask=mask
        )


_TL_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


class LinearStateCommit:
    """Moves the accepted rows of a tree round into chain order in GDN / Mamba2 layers, all layers in one launch each.

    ``slots[i]``: layer ``i``'s KV cache group's persistent state-slot buffer (one state slot per verify row).
    """

    def __init__(self, layers: Sequence[Any], slots: Sequence[torch.Tensor], device: torch.device):
        self.num_layers = len(layers)
        if not self.num_layers:
            return
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        def window(layer: Any) -> torch.Tensor:
            cache = layer.kv_cache[0]
            return cache if is_conv_state_dim_first() else cache.transpose(-1, -2)

        first_conv, first_ssm = window(layers[0]), layers[0].kv_cache[1]
        for layer in layers:
            conv, ssm = window(layer), layer.kv_cache[1]
            if (
                conv.stride() != first_conv.stride()
                or conv.dtype != first_conv.dtype
                or ssm.stride() != first_ssm.stride()
            ):
                raise ValueError("the linear-attention layers must share one state layout")
        self.conv_dtype, self.ssm_dtype = _TL_DTYPES[first_conv.dtype], _TL_DTYPES[first_ssm.dtype]
        self.conv_strides = first_conv.stride()
        self.dim = first_conv.shape[1]
        self.ssm_stride, self.ssm_numel = first_ssm.stride(0), first_ssm[0].numel()
        self.conv_ptrs = torch.tensor([window(layer).data_ptr() for layer in layers], dtype=torch.uint64, device=device)
        self.ssm_ptrs = torch.tensor(
            [layer.kv_cache[1].data_ptr() for layer in layers], dtype=torch.uint64, device=device
        )
        self.slots_ptrs = torch.tensor([s.data_ptr() for s in slots], dtype=torch.uint64, device=device)
        self.slots_strides = torch.tensor([s.stride(0) for s in slots], dtype=torch.int64, device=device)
        if any(s.dtype != torch.int32 for s in slots):
            raise ValueError("state slots must be int32")

    def commit(self, idx_mapping: torch.Tensor, path: torch.Tensor, path_len: torch.Tensor) -> None:
        if not self.num_layers:
            return
        num_reqs, num_layers = idx_mapping.shape[0], self.conv_ptrs.shape[0]
        block = 128
        _compact_window_kernel[(num_reqs, num_layers, triton.cdiv(self.dim, block))](
            self.conv_ptrs,
            self.slots_ptrs,
            self.slots_strides,
            idx_mapping,
            path,
            path.stride(0),
            path_len,
            *self.conv_strides,
            self.dim,
            BLOCK=block,
            DTYPE=self.conv_dtype,
            num_warps=1,
        )
        block = 1024
        _move_state_kernel[(num_reqs, num_layers, triton.cdiv(self.ssm_numel, block))](
            self.ssm_ptrs,
            self.slots_ptrs,
            self.slots_strides,
            idx_mapping,
            path,
            path.stride(0),
            path_len,
            self.ssm_stride,
            self.ssm_numel,
            BLOCK=block,
            DTYPE=self.ssm_dtype,
            num_warps=4,
        )
