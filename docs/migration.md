# Migrating existing EAGLE-4 configurations

Eagle-harness reads JSON configuration, not these environment variables. Use the field in the table inside
`--speculative-algorithm-config` (sglang) or `method_config` (vLLM). See [Configuration](configuration.md) for supported
engine paths and defaults. A mapped field does not imply both engines implement it.

Dotted paths describe nested JSON objects, not literal keys. For example, `cost_model.chain_tiny` becomes
`{"cost_model": {"chain_tiny": [2.60, 0.24, 0.00]}}`. Use JSON booleans for switches and arrays for coefficient
triples.

`Dropped` means there is no corresponding option: the switch controlled a fixed optimization, a removed kernel,
or diagnostic output. The defaults below describe the legacy environment-variable configuration, not an
environment parser.

## Streamlined draft fields

Remove `draft_model.moe_top_k` and `draft_model.packed_candidate_gather` from old JSON; the loader rejects them
as unknown keys. Expert counts come from the checkpoint. The internal `CostModel.ragged_recent_verify_nums`
state is also removed.

## Removed in 0.1.1

Streamlined configuration: these options are removed, and serving behaves the same without them. A config that
still sets one stops at load with `<option>: removed in eagle-harness 0.1.1`. Delete the option; use
`"mode": "serve"` in place of `"eval"`.

- `"mode": "eval"`
- `ragged` (`enabled`, `force_uniform`)
- `cost_model.ragged_draft_relax`, `cost_model.ragged_verify_threshold`
- `draft_model.moe_gate_acc` (SwiGLU-MoE drafts)

## Environment variables

| Environment variable | JSON field | Legacy default | Conversion |
|---|---|---|---|
| `SGLANG_EAGLE_HARNESS_CHAIN_TINY` | `cost_model.chain_tiny` | `2.60,0.24,0.00` | Set the field |
| `SGLANG_EAGLE_HARNESS_CHAIN_SMALL` | `cost_model.chain_small` | `2.85,0.24,0.00` | Set the field |
| `SGLANG_EAGLE_HARNESS_CHAIN_MEDIUM` | `cost_model.chain_medium` | `2.54,0.24,0.70` | Set the field |
| `SGLANG_EAGLE_HARNESS_CHAIN_LARGE` | `cost_model.chain_large` | `1.55,0.24,0.00` | Set the field |
| `SGLANG_EAGLE_HARNESS_CHAIN_SCALE_TINY` | `cost_model.chain_scale_tiny` | `1.25` | Set the field |
| `SGLANG_EAGLE_HARNESS_CHAIN_TINY_MAX_BS` | `cost_model.chain_tiny_max_bs` | `2` | Set the field |
| `SGLANG_EAGLE_HARNESS_CHAIN_SCALE_SMALL` | `cost_model.chain_scale_small` | `1.45` | Set the field |
| `SGLANG_EAGLE_HARNESS_CHAIN_SCALE_MEDIUM` | `cost_model.chain_scale_medium` | `2.00` | Set the field |
| `SGLANG_EAGLE_HARNESS_RAGGED_DRAFT_RELAX` | — | `0.75 (all three)` | Dropped (removed in 0.1.1) |
| `SGLANG_EAGLE_HARNESS_RAGGED_RELAX_SMALL` | — | `0.75` | Dropped (removed in 0.1.1) |
| `SGLANG_EAGLE_HARNESS_RAGGED_RELAX_MEDIUM` | — | `0.75` | Dropped (removed in 0.1.1) |
| `SGLANG_EAGLE_HARNESS_RAGGED_RELAX_LARGE` | — | `0.75` | Dropped (removed in 0.1.1) |
| `SGLANG_EAGLE_HARNESS_RAGGED_VERIFY_SMALL` | — | `1.35` | Dropped (removed in 0.1.1) |
| `SGLANG_EAGLE_HARNESS_RAGGED_VERIFY_MEDIUM` | — | `1.60` | Dropped (removed in 0.1.1) |
| `SGLANG_EAGLE_HARNESS_RAGGED_VERIFY_LARGE` | — | `1.40` | Dropped (removed in 0.1.1) |
| `SGLANG_EAGLE_HARNESS_VERIFY_RAGGED` | — | `1` | Dropped (removed in 0.1.1) |
| `SGLANG_EAGLE_HARNESS_RAGGED_FORCE_UNIFORM` | — | `0` | Dropped (removed in 0.1.1) |
| `SGLANG_EAGLE4_MOE_TOP_K` | — | `unset` | Dropped; use checkpoint `num_experts_per_tok` |
| `SGLANG_EAGLE4_MOE_GATE_ACC` | — | `fp32` | Dropped (removed in 0.1.1) |
| `SGLANG_EAGLE4_PACKED_CANDIDATE_GATHER` | — | `1` | Dropped; the implementation chooses the gather path |
| `SGLANG_EAGLE4_CUDA_CANDIDATE_KERNELS` | `draft_model.cuda_candidate_kernels` | `1` | Set the field |
| `SGLANG_EAGLE_DRAFT_EMBED_FULL` | `draft_model.embed_full` | `1` | Set the field |
| `SGLANG_EAGLE4_REPLICATED_DRAFT_EMBED` | `draft_model.embed_full` | `1 (deprecated alias)` | Set the field |
| `SGLANG_DISABLE_EAGLE_DRAFT_EMBED_TRITON` | `draft_model.embed_triton` | `0` | Invert the boolean |
| `SGLANG_DISABLE_EAGLE_HARNESS_VERIFY_ARGMAX` | — | `0` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_SAMPLED_TREE` | — | `0` | Removed |
| `SGLANG_DISABLE_FLASHINFER_VERIFY_EMPTY_OUT` | — | `0` | Removed |
| `SGLANG_DISABLE_FLASHINFER_VERIFY_TAIL_MASK` | — | `0` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_FUSED_VERIFY_PREP` | — | `0` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_VERIFY_REPLAY_TRIMS` | — | `0` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_FOLD_COMMIT_STEPS` | — | `0` | Removed |
| `SGLANG_EAGLE_HARNESS_DRAFT_GRAPH_OPT` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_DRAFT_STRIDED_Q` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_DRAFT_ATTN_OUT` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_DRAFT_NORM_HALVES` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_DRAFT_GATHER_DOT_BK1` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_EXTEND_LAST_ROW_GATHER` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_EXTEND_FC_NORM_THIRDS` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_EXTEND_FC_STRIDED_INPUT` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_EXTEND_STRIDED_Q` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_FUSED_STEP_UPDATE` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_FUSED_STEP_ZERO_COPY` | — | `1` | Removed |
| `SGLANG_EXACT_SOFTMAX` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_SIDE_STREAM_VERIFY_SCORES` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_REUSE_EXTEND_DRAFT_TOPK` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_CHECK_EXTEND_DRAFT_TOPK` | — | `0` | Removed |
| `SGLANG_EAGLE_HARNESS_FUSED_POST_VERIFY` | — | `1` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_DRAFT_PRELAUNCH` | — | `0` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_SIDE_STREAM_STATE_COMMIT` | — | `0` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_LEAN_DRAFT_INIT` | — | `0` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_LEAN_DRAFT_STEP` | — | `0` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_SKIP_NOOP_MOE_CONTEXTS` | — | `0` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_DRAFT_STEP_META` | — | `0` | Removed |
| `SGLANG_DISABLE_EAGLE_HARNESS_DRAFT_PREP_ONE_LAUNCH` | — | `0` | Removed |
| `SGLANG_EAGLE_HARNESS_VERIFY_PREALLOC` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_CACHED_VERIFY_BATCH` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_DEVICE_ACCEPT` | — | `1` | Removed |
| `SGLANG_EAGLE_HARNESS_DEVICE_UPDATE_TREE` | — | `1` | Removed |
| `SGLANG_OPT_GDN_TREE_GLUON` | — | `1` | Removed |
| `SGLANG_OPT_GDN_TREE_GLUON_PDL` | — | `1` | Removed |
| `SGLANG_OPT_GDN_TREE_GLUON_V2` | — | `1` | Removed |
| `SGLANG_OPT_GDN_TREE_GLUON_V3` | — | `1` | Removed |
| `SGLANG_DISABLE_GDN_TREE_VERIFY_IN_PLACE` | — | `0` | Removed |
| `SGLANG_EAGLE_HARNESS_CHAIN_DEBUG` | — | `0` | Removed |
| `SGLANG_EAGLE_HARNESS_CHAIN_HIST` | — | `unset` | Removed |
| `SGLANG_EAGLE4_SWA_AUDIT` | — | `unset` | Removed |
| `SGLANG_EAGLE4_SWA_AUDIT_DIR` | — | `unset` | Removed |
| `DRAFT_TREE_PATH` | `data_path` | `--speculative-data-path` | Set the field |
| `COMPUTE_INTENSITY` | `intensity` | `--speculative-compute-intensity` | Set the field |
| `DEBUG_STEP` | — | `0` | Removed |

## Command-line arguments

| Legacy argument | JSON field |
|---|---|
| `--speculative-exec-mode` | `mode`: `-1` → `static`, `0` → `serve`, `1` → `eval`, `2` → `calibrate` |
| `--speculative-data-path` | `data_path` |
| `--speculative-compute-intensity` | `intensity` |
| `--speculative-algorithm EAGLE4` | `draft_family`: `eagle4` |

Use `--speculative-algorithm EAGLE_HARNESS` on sglang or `method: "eagle_harness"` on vLLM to select the plugin.
