# Architecture

eagle-harness adds one speculative decoding method to an existing serving engine. The engine keeps prefill,
scheduling, KV cache, sampling and the HTTP server. eagle-harness supplies the draft tree, the cost model that sizes
it, the EAGLE-4 draft model and the kernels of the tree round. It never edits or patches the engine: it plugs in
through the engine's plugin entry point and a set of generic hooks the engine exposes.

## How the plugin loads

Each engine loads installed plugins in every process at startup, before it resolves the model config.

| engine | entry point group | function | what it registers |
|---|---|---|---|
| sglang | `sglang.srt.plugins` | `eagle_harness.sglang.plugin:register` | the speculative algorithm `EAGLE_HARNESS` and the draft class `LlamaForCausalLMEagle4` |
| vLLM | `vllm.general_plugins` | `eagle_harness.vllm.plugin:register` | the speculative method `eagle_harness`, the draft architecture `LlamaForCausalLMEagle4` and the tree attention backend `CUSTOM` |
| TokenSpeed | `tokenspeed.plugins` | `eagle_harness.tokenspeed.plugin:register` | the drafter `EAGLE_HARNESS` (reading the target's aux hidden states) and the draft architecture `LlamaForCausalLMEagle4` |

Before it registers anything, the plugin checks that every hook it uses is present in the installed engine. The
contract lives in code: `eagle_harness/sglang/hooks.py` lists every sglang attribute the plugin overrides or calls,
with its parameter names, `eagle_harness/vllm/adapter.py` (`HOOK_MAP`) lists every vLLM hook by name, and
`eagle_harness/tokenspeed/hooks.py` lists every TokenSpeed name and parameter the adapter relies on. If one is
missing or changed, the plugin registers nothing and logs one error naming the engine version, each mismatch and the
fix. The engine keeps working without the method. [Compatibility](compatibility.md) has the version rules and
[Troubleshooting](troubleshooting.md) shows the messages.

## The hooks each engine provides

sglang runs the method on its V2 speculative worker with either overlap or synchronous scheduling.
`--disable-overlap-schedule` selects synchronous scheduling. The plugin uses these hooks of
sglang's `CustomSpecAlgo` and related classes:

| hook | what eagle-harness does with it |
|---|---|
| `handle_server_args` | parses the method's JSON config, checks the tree shape and the hardware, turns off mixed chunked prefill |
| `supports_mm_inputs` | refuses multimodal inputs (sglang's `validate_request` gives the reason) |
| `req_to_token_extra_len` | reserves block-table room for the largest tree |
| `create_decode_graph_variants` | captures one target-verify CUDA graph per verify width |
| `create_linear_attn_verify_extension` | runs the GDN tree verify kernels in hybrid (gated delta net) targets, and the tree conv1d in Mamba2 hybrids (NemotronH) |
| `SpecInput.flashinfer_verify_variant` | gives FlashInfer a tree-tail mask attention variant for batch-size-1 verify graphs |
| `accept_step` of `run_eagle_verify` | replaces the accept bookkeeping of a greedy batch-size-1 round with two fused launches |

vLLM runs the method on its Model Runner V2. The plugin uses its speculative-method registry, the speculator hooks
`prepare_verify_inputs`, `make_rejection_sampler` and `post_accept`, the linear-attention verify extension for GDN
layers, and the `CUSTOM` attention backend slot.

TokenSpeed runs the method as a drafter on its plugin registries. The plugin uses `register_drafter` and
`register_model` from TokenSpeed main and generic hooks of the eagle-hq TokenSpeed build: the method's JSON
config (`--speculative-algorithm-config`, delivered to `BaseDrafter.configure_algorithm`), the target's aux hidden
states for a plugin algorithm, and tree drafting by a plugin drafter. The same branch lets a plugin draft bind its
cache under a hybrid target and adds sliding-window attention to tree lanes. See [TokenSpeed](#tokenspeed) below.

## One decode round

Every decode step of a request is a round: draft a tree, verify it with one target forward, accept a path, then
prepare the next round.

1. The EAGLE-4 draft model drafts: it grows a token tree one level at a time from the last accepted token. Each step
   expands the best `K` frontier nodes (`--speculative-eagle-topk` on sglang, `node_top_k` on vLLM). The draft reads
   three hidden states of the target, attends to its own sliding window plus each row's tree ancestors, and scores
   the vocabulary with a low-rank candidate head: a rank-r estimate picks a few candidates per vocabulary shard, and
   only those are re-scored exactly with the full LM head rows.
2. The tree is ranked: a node's score is its parent's score times its probability. The verify set is the best
   `width - 1` nodes plus the root, ordered parent first, so any prefix of it is a tree too.
3. The cost model sizes the round. On sglang it decides, after every second draft level, whether another level pays for
   its draft time, and picks the round's verify width from the measured verify cost of each width. It reads the
   [cost tables](calibration.md) of the batch size and context length at hand. On vLLM the scheduler fixes the verify
   width per batch size, so the per-round decision is the draft depth: a step runs only while a deeper node could
   still enter the verify set, and, with cost tables, only while its expected gain pays for its measured cost.
4. One target forward verifies the tree: it scores every node of the verify set. Full-attention layers attend each row to the
   committed context and to its own ancestors (a tree mask). GDN layers run the tree versions of the causal conv1d
   and the gated delta rule recurrence, each node continuing from its parent's state.
5. The target's token at each node decides which child, if any, the walk follows from the root. For a greedy
   request it is the target's argmax. For a sampled request it is a draw from the target's distribution at that
   node (temperature, top-k, top-p, min-p): on vLLM the draw vLLM's sampler makes at that position, with noise
   keyed by the request's seed and the node's depth; on sglang, sglang's tree speculative sampling. The accepted
   path plus one bonus token from the target are committed. Every committed token is the target's own token given
   the accepted prefix, so the output has the target's distribution. Greedy output matches greedy decoding without
   speculation, up to near-ties in the target's logits that a different batch shape can flip. The accepted rows'
   KV and GDN states are moved to the request's chain positions.
6. The draft extend runs the draft over the accepted tokens, which updates its KV cache and yields the next round's
   first candidates.

On sglang the round runs as CUDA graph replays: one graph per draft step, one per verify width on the target, one for
the draft extend and small graphs for ranking and building the verify input. The host only reads the cost model's
inputs, one step late, so it rarely waits for the GPU. While the scheduler processes a round, the next round's
draft extend and first draft steps already run. Synchronous scheduling uses a side stream. Overlap scheduling
uses the forward stream so the engine can relay the draft outputs in order.

## TokenSpeed

The TokenSpeed adapter is experimental. The paths below describe implementation, not qualification for every workload.

TokenSpeed's tree attention, GDN tree verification, accept walk and accepted-path compaction serve both adapter modes.
In static mode the target forward, acceptance and all draft steps replay as one decode CUDA graph.
eagle-harness supplies the tree policy and the draft model:

- `HarnessDrafter` is TokenSpeed's EAGLE tree drafter with EAGLE-4's seed, expand and finalize steps. The draft
  extend and the `steps - 1` lane forwards of `K` rows per request are TokenSpeed's.
- The EAGLE-4 draft model (`LlamaForCausalLMEagle4`) uses a low-rank candidate head. Its exit returns candidate-head
  rows without logits, so no draft step runs the whole-vocabulary LM head.
- Static mode uses the shared core draft tree to score nodes, keep the frontier and pick verify nodes. Given the
  same draft outputs, it selects the same tree as the sglang and vLLM adapters
  (`tests/test_tokenspeed_adapter.py`). `eagle_harness/tokenspeed/tree_lanes.py` turns the frontier choices into
  TokenSpeed's 64-bit lane masks (each lane sees its ancestors' draft slots) and the verify window and parents.

The EAGLE-4 draft layer has a sliding window. TokenSpeed main refuses a sliding-window layer in its tree lanes; the
hooks branch adds that path (every lane of a request sits at one position, so all lanes see the same committed keys).

Serve mode captures the target, sampling and draft extend together. The host launches the first draft-depth graph.
The GPU cost policy decides whether to continue and tail-launches later depths and finalization. Calibration parsing,
cost adjustments and depth thresholds come from the shared core cost model. `CostModel.tree_policy_costs()` exports
immutable host cost records once at binding. The installed TokenSpeed engine packs these records and owns the
device policy, draft-round graphs, tree kernels and tail-launch runtime; those implementations are not vendored in
eagle-harness. Its GPU policy consumes the shared thresholds without a second host cost model.

For trees, the physical verify width stays fixed at `--speculative-num-draft-tokens`. A round can select fewer meaningful nodes
and pad the rest with unconnected roots. Draft lane slots `(steps - 1) * K` have their own KV reservation and must fit
in a 64-bit lane mask, rather than in the verify width. Per-round lane page-table snapshots isolate deferred work
and direct padded rows to null pages. Round completion fences protect request-slot reuse and cache mutation.

## Package layout

| path | contents |
|---|---|
| `eagle_harness/core/` | cost model, cost tables, calibration fits, draft tree bookkeeping, `HarnessConfig` (pure Python, tested on CPU) |
| `eagle_harness/kernels/` | Triton, Gluon and native CUDA kernels, hardware dispatch in `arch.py` (no engine imports) |
| `eagle_harness/models/` | EAGLE-4 draft config validation and the low-rank head |
| `eagle_harness/sglang/` | sglang plugin, algorithm hooks, workers, draft steps, verify graphs, draft model |
| `eagle_harness/vllm/` | vLLM plugin, speculator, tree drafter, tree attention backend, GDN tree verify, tree accept, draft model |
| `eagle_harness/tokenspeed/` | TokenSpeed plugin, hook check, drafter (tree policy), lane masks and verify window, draft model |
| `eagle_harness/compat.py` | the supported engine versions and the hook check behind `eagle-harness-check` |
