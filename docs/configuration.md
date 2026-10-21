# Configuration reference

The method takes one JSON object. On sglang it goes in `--speculative-algorithm-config`, inline or as a path to a
JSON file. On vLLM it goes in the `method_config` of `--speculative-config`. On TokenSpeed it goes in
`--speculative-algorithm-config` (inline JSON). The adapters share the core configuration parser and add
engine-specific keys and restrictions described below. Unknown keys are
rejected with the list of valid keys, so a typo cannot fall back to a default silently.

Source of truth: `eagle_harness/core/config.py` (`HarnessConfig` and its parts), `eagle_harness/vllm/config.py`
(the vLLM-only keys) and `eagle_harness/tokenspeed/config.py` (what TokenSpeed accepts).
For legacy environment variables and CLI flags, see [Migration](migration.md).

## Top-level keys

| key | type | default | meaning |
|---|---|---|---|
| `mode` | string | `"serve"` | `serve`: the cost model sizes every round. `calibrate`: reserved for cost measurement (see [Calibration](calibration.md)). `static`: the fixed tree of the shape flags every round, no cost model (on SGLang the same EAGLE-4 round, graphs and kernels as the cost-model modes). |
| `data_path` | string | none | Offline calibration directory whose `cost/` subdirectory holds the cost tables and whose `key.json` names the setup they were measured on (GPU, TP size, model pair, engine, dtype); a directory keyed for another setup is refused. On SGLang, omitting it uses a matching packaged profile when available, otherwise live served-round costs ([Calibration](calibration.md#costs-measured-on-the-served-rounds)). vLLM and TokenSpeed require the engine-specific inputs below. |
| `cost_cache` | string | `~/.cache/eagle-harness/costs` | Directory of the live-cost cache files (one per setup key). |
| `intensity` | string | `"high_intensity"` | Draft shape family of the cost tables: `high_intensity`, `low_intensity` (trees) or `chain_intensity` (one candidate per step). It sets the batch limit and the verify widths that get a CUDA graph. |
| `draft_family` | string | `"eagle4"` | `eagle4`. Sets the cost model's depth loss when `cost_model.depth_loss_ratio` is not given. |
| `cost_model` | object | see below | Cost model constants. |
| `draft_model` | object | see below | EAGLE-4 draft model options. |
| `l2_warm` | object | see below | L2 prefetches of target weights inside the sglang verify graphs. |

## `cost_model`

Dimensionless constants of the cost model's stopping rules. The defaults are the values the cost tables were fitted
with. Change them only together with a measurement.

| key | type | default | meaning |
|---|---|---|---|
| `threshold` | number | `2.5` | Gain per unit cost a tree level must reach to be drafted. Higher stops trees earlier. |
| `async_layer_ratio` | number | `0.9` | Factor on a level's gain per unit cost before it is compared with the bar. |
| `depth_loss_ratio` | number or `null` | `null` | Raises the bar by this fraction per depth level. `null` derives it: `0.04` for `eagle4`, doubled for MoE targets (chains keep the base value). |
| `chain_tiny`, `chain_small`, `chain_medium`, `chain_large` | 3 numbers | `[2.60, 0.24, 0.00]`, `[2.85, 0.24, 0.00]`, `[2.54, 0.24, 0.70]`, `[1.55, 0.24, 0.00]` | Per chain size class: threshold, depth loss, cost power. |
| `chain_scale_tiny`, `chain_scale_small`, `chain_scale_medium` | number | `1.25`, `1.45`, `2.00` | Bounds on the target's verify cost ratio that separate the chain size classes. |
| `chain_tiny_max_bs` | integer | `2` | Largest batch size of the tiny chain class. |

On vLLM, served-chain widths come from the measured costs and acceptance profile described below, not from
`cost_model.chain_*`. Options removed in 0.1.1 are listed in [Migration](migration.md#removed-in-011).

## `draft_model`

| key | type | default | meaning |
|---|---|---|---|
| `cuda_candidate_kernels` | boolean | `true` | Use the native CUDA candidate kernels where the GPU supports them. With `false` the Triton kernels run (same results). |
| `embed_full` | boolean | `true` | Keep the full draft embedding table on every rank, so a draft step looks tokens up without an all-reduce. |
| `embed_triton` | boolean | `true` | Look draft embeddings up with sglang's Triton kernel (needs the full table on the rank). |

The supported latent-MoE draft uses the host engine's expert implementation.

## `l2_warm` (sglang)

Each entry is `[share, ctas]`: prefetch the last `share` (0 to 1) of a weight's bytes into L2 with `ctas` thread
blocks on a side stream, or `null` for off. The prefetches only read memory, so results do not change.

| key | default | what is warmed, and when |
|---|---|---|
| `gate_up_gdn` | `[0.42, 16]` | the MLP gate/up weight, after a GDN layer's input projection |
| `gate_up_attention` | `[0.3, 74]` | the MLP gate/up weight, after an attention layer's QKV projection |
| `down_proj` | `null` | the MLP down weight, after the mixer's output projection |
| `next_input` | `null` | the next layer's input projection, after the MLP's down projection |

## Engine arguments

sglang reads the largest tree shape from its own server arguments, and all three are required:

| sglang argument | meaning |
|---|---|
| `--speculative-num-steps` | maximum draft depth |
| `--speculative-eagle-topk` | frontier nodes expanded per step (at most 8) |
| `--speculative-num-draft-tokens` | maximum verify width, root included |

In the cost-model modes each round's depth and width are chosen within that shape.

vLLM takes these extra keys in `method_config`:

| key | type | default | meaning |
|---|---|---|---|
| `max_verify_width` | integer | `32` (trees), `16` (chains) | Largest verify width, root included. vLLM's `num_speculative_tokens` is the widest verify the batch-size schedule runs minus one (this minus one without a schedule): vLLM keeps that many extra linear-attention state slots per request. Set only one of the two. |
| `node_top_k` | integer | `8` (trees), `1` (chains) | Frontier nodes expanded per draft step. `1` uses vLLM's own EAGLE chain loop. |
| `tree_depth` | integer | `12` (trees), `15` (chains) | Maximum draft depth. It sizes the draft's KV lookahead. |
| `depth_stop_gain` | `"derived"`, number or `null` | `"derived"` | What one more draft step must add to the expected accepted length. `"derived"`: the step's measured cost from the cost tables under `data_path`, or a startup error in automatic serving when a scheduled shape is unpriced. Static mode retains a lossless stop. A number sets a fixed gain. `0` stops only when no deeper node can enter the verify set. `null` always drafts `tree_depth` steps. |
| `record_rounds` | boolean | `false` | Trees: time the served decode rounds and merge them into `<data_path>/rounds/tree.json`, per batch size and verify width: the median round period and the mean tokens committed per request-round, as vLLM's spec-decode counters report it (rounds that also prefill included). Needs `data_path`. Serve the workload once per candidate `max_verify_width`; each server keeps the other widths' rows. |

In `serve` mode vLLM fixes the verify width per batch size (`num_speculative_tokens_per_batch_size`).
Automatic trees require cost tables for every scheduled shape; set `depth_stop_gain` explicitly to opt out.
Automatic chains require timings and an acceptance profile; an explicit `max_verify_width` or batch schedule opts into unpriced operation.
Trees get the widest graph width of their cost buckets, capped from the first batch size where
`<data_path>/rounds/tree.json` measured a narrower verify faster (as in static mode). A chain (`node_top_k` 1) verifies exactly its width, so the
width of each batch size is derived from the cost tables and the chain acceptance profile under `data_path`
(`<data_path>/accept/chain.json`): the width with the most expected tokens per unit of time, `1` (no speculation) when
no chain pays. Without a profile startup fails, unless `max_verify_width` or a batch schedule is explicitly supplied. See
[calibration](calibration.md#chain-widths).

Chains on vLLM: leave the target on vLLM's default attention backend and give only the draft FlashAttention,
`"attention_backend": "FLASH_ATTN"` next to `"method"` in `--speculative-config`. FlashInfer cannot serve the
multi-head EAGLE-4 draft on the 128-token pages a hybrid target needs, and `--attention-backend FLASH_ATTN` moves the
whole target off FlashInfer. Keep the target backend unchanged for a matched speculation comparison.

TokenSpeed is experimental. It takes the draft shape from its server arguments. `"mode": "serve"` is the default, as on
the other engines: it uses calibrated GPU-side decisions to stop drafting and select meaningful verify nodes, and needs
`data_path` unless a complete `chain_width_table` (from batch size 1) sets every width. Without a usable config the
server refuses to start. `"mode": "static"` uses every configured draft step and needs no cost data.
Tree modes keep one physical verify width per server. Serve-mode trees pad unused nodes with unconnected roots.
Pass `--sampling-backend greedy` because the default FlashInfer sampler cannot verify draft trees.

For calibrated tree serving, provide `data_path` containing the calibration `cost/` directory.
The [TokenSpeed quickstart](quickstart-tokenspeed.md) distinguishes the single-client cap of 4 from the
concurrent cap of 32. Neither is a general performance recommendation.
The calibration must match the target, draft, hardware, tensor-parallel setup and recurrent-state dtype.
`calibrate` and live costs are not supported by this adapter. Chains use top-k 1 and `chain_intensity`.
An explicit `chain_width_table` contains increasing `[minimum_batch_size, verify_width]` pairs in `serve` mode.
A complete schedule starting at batch size 1 does not need cost files. This describes the configuration interface,
not a measured high-concurrency speed claim. Sampled-tree distribution quality is unqualified.

| TokenSpeed argument | Meaning |
|---|---|
| `--speculative-num-steps` | Maximum draft depth, subject to the installed engine's shape checks |
| `--speculative-eagle-topk` | Frontier nodes expanded per step, 1 for a chain |
| `--speculative-num-draft-tokens` | Physical verify width, root included, large enough for the selected top-k |

The lane window `(steps - 1) * topk` must fit in 64 slots. Its KV reservation is independent of the verify width,
so depth 12, top-k 4 and verify width 16 are supported in serve mode. Static mode also encodes a
depth-first sort key and supports at most ten steps. This is distinct from the lane-window limit.

## Shape limits

From `eagle_harness/core/limits.py`. A shape outside them is refused at startup with the reason.
SGLang and vLLM share the tree batch limit. SGLang retains its memory-conservative default of 32 tree requests
(48 for chains) when `max_running_requests` is omitted. Explicit larger limits are admitted.

| limit | trees | chains |
|---|---|---|
| running requests | 512 | 512 |
| draft depth | 31 | 15 |
| verify width | 64 | 16 |
| frontier nodes per step | 8 | 1 |
