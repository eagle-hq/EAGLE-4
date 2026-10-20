---
type: how-to
owner: eagle-harness
last_verified: 2026-10-20
---

# Calibrate your serving setup

## Costs measured on the served rounds

On SGLang the cost-model modes need no offline step. Without `data_path`, a matching packaged MI355X profile
is selected when available. Otherwise, the server prices the draft step and target verify on its own served rounds. The first rounds at each batch bucket (a captured graph batch size) force
every captured verify width a few times, timed with CUDA events around the exact kernels served at the real
context length; after that every round refines its cell (an exponential moving average per batch bucket, verify
width and 1024-token context interval) and the cost model reads the resulting tables. The cells are cached under
`cost_cache` (default `~/.cache/eagle-harness/costs/<key>.json`), keyed by GPU name, TP size, the target's and the
draft's `config.json` digests, the engine version and the dtype, so a restart on the same setup starts with the
costs it ended with and a different setup measures again. All TP ranks decide from rank 0's cells.

The live cache also checks the draft top-k and depth, because they change the work being timed. Invalid or
unreadable caches are ignored and remeasured. Cached and refreshed curves outside the cost model's supported
ratios keep the fixed-shape policy until usable measurements arrive; they do not fail the next decode round.
Seeding is bounded by the captured widths and sample count, not a wall-clock deadline, and occurs on served
rounds (including engine warmup requests), not during graph capture. Live timings alone do not establish
optimal acceptance or no-loss behavior at high concurrency; compare sustained workloads against native no-spec.
After seeding, a draft prefix is timed once per table-refresh period; target timings continue on served verifies.
Prelaunched draft samples use that batch's bucket and context, not the preceding batch's seed key.

Offline calibration below remains the precise mode: it times isolated steps on dummy batches over a fine context
grid. Its tables carry the same key in `key.json`; a server refuses tables keyed for another setup, and accepts
tables without a key (measured before keys existed) with a warning.

## Offline calibration

Use `eagle-harness calibrate` to measure costs on the GPUs and engine you will serve with. It starts and stops
its own engine processes, retains raw measurements and prints a matching serving command. Stop other servers
on your allocated GPUs first.

## Run the command

The released offline calibration path is vLLM. SGLang served-chain calibration requires the host graph
contract with per-variant forward modes because it also measures native decode. The pinned SGLang CUDA
release engine does not expose that contract, so the CLI's SGLang calibration path fails with an explicit
error on that build. Use the [SGLang live-cost serving recipe](quickstart-sglang.md) without an offline step.
The SGLang commands below document the driver interface for a host with the required contract, not a
working calibration recipe on the pinned release engine.

Set `MODEL`, `DRAFT` and `TP` to your target, compatible exported draft and tensor-parallel size:

```bash
eagle-harness calibrate --engine vllm --model "$MODEL" --draft "$DRAFT" --tp "$TP"
```

Without `--engine`, the CLI selects the sole installed engine. If both engines are installed, specify `--engine sglang` or
`--engine vllm`. TP defaults to one. No private dataset or calibration directory is required.

| Engine | Default measurement | Default output |
|---|---|---|
| vLLM | Batch-one tree, verify widths 8, 16, 32 and 64, with context lengths below 4096 | `eagle-calibration-vllm/` |
| SGLang | Served chains up to width 8, at concurrency 1 and 8, with 1024 new tokens per request | `eagle-calibration-sglang/` |

The package includes the SGLang served-chain calibration driver. It requires the host graph contract above
and does not produce dynamic-tree cost tables. vLLM prices isolated whole steps and draft graph replays, not a full high-concurrency served round.
Neither command establishes a speedup or replaces a correctness check against no-spec generation.

Preview the settings without importing an engine, loading a model or creating output:

```bash
eagle-harness calibrate --engine vllm --model "$MODEL" --draft "$DRAFT" --tp "$TP" --dry-run
```

Use `--output` to choose a fresh directory. The command refuses to overwrite one, including partial results
from a failed run. GPU allocation, model downloads and compilation can make a calibration take much longer
than five minutes.

## Use representative SGLang prompts

The default built-in prompts are a command smoke test. Acceptance depends on the workload, so use held-out
requests that resemble your traffic for a deployable profile. A JSONL file uses the SPEED-Bench first-turn format:

```json
{"category":"coding","turns":["Explain a database transaction with an example."]}
{"category":"writing","turns":["Write a short science lesson about rainfall."]}
```

Only the first turn is used. The target tokenizer applies its chat template. Prompts are repeated if a
concurrency level needs more requests than the file contains.

```bash
eagle-harness calibrate --engine sglang --model "$MODEL" --draft "$DRAFT" --tp "$TP" \
  --data heldout.jsonl --batch-sizes 1,8 --output calibration-workload
```

The driver records served round periods and the widest chain's accepted-token counts, then reports the chosen
width by batch size. Width one verifies only the root and still extends draft KV. Width zero runs native target
decode without draft forward, verification or prelaunch. Calibration measures it last and keeps affected requests
OFF until they finish. A mixed batch containing an OFF request stays OFF, so no request resumes with stale draft KV.

Native OFF requires DP1 and disabled Mamba prefix-cache tracking. The generated command disables radix caching.
It retains the synchronous EAGLE-4 scheduler and the loaded draft model, so it is not guaranteed to match a
separately started no-spec engine. Compare measured throughput against that engine before deployment. These are
measured choices for the supplied workload, not a universal no-loss guarantee.

## Match the serving settings

Use `--engine-args` for settings that affect kernel time, such as vLLM's linear backend:

```bash
eagle-harness calibrate --engine vllm --model "$MODEL" --draft "$DRAFT" --tp "$TP" \
  --engine-args '{"linear_backend":"flashinfer_cutedsl"}' --output calibration-cutedsl
```

Keys are engine Python keyword names with underscores. The command refuses overrides of the model, draft,
TP, process controls or speculative configuration. The same permitted settings go into the generated server
command. vLLM calibration supports only batch size one through this command.

A successful run writes `settings.json` and `serve.sh` alongside the measurements. Start the measured setup with:

```bash
sh calibration-workload/serve.sh
```

Use the output directory from your own run in that command. The server binds to loopback by default.
Move the generated command to your service configuration before deploying it, and configure authentication
or a private proxy before exposing the API.

## Read the artifacts

| Path below the output directory | Contents |
|---|---|
| `settings.json` | Model identifiers, TP, engine, batch sizes and engine overrides |
| `raw/width-*.json` | vLLM whole-step timings |
| `cost/bs_1/model_llm/` | Fitted vLLM target verify costs |
| `cost/bs_1/model_slm/` | Fitted vLLM draft costs |
| `config/measure_config.json` | Fit grid and measurement parameters |
| `raw/sglang_rounds.json` | SGLang served round observations |
| `rounds/chain.json` | SGLang chain round costs |
| `accept/chain.json` | SGLang accepted counts per draft position |
| `*.log` | Engine startup, measurement and fit logs |
| `serve.sh` | Matching server command, written only after success |

The vLLM sweep takes `context_stop` from the resolved `max_model_len` (including engine overrides),
records it in each raw file, and fits that same range. The default engine limit remains 4608.
The limit must be a multiple of the context step, with room for the widest verify after the
last sample; unsupported grids fail rather than silently measuring a shorter context.
Raw files without `context_stop` must be regenerated. Context-indexed cost tables log once
per table when a request exceeds the calibrated range and the last interval is reused.

The core fit and file schema live in `eagle_harness/core/calibration.py` and `cost_table.py`.
The shared context-indexed cost model selects the nearest calibrated batch size, with ties going to the larger
size, and clamps context lengths to the measured range. The generated vLLM static-tree command instead uses
median costs across context intervals for depth stopping. It does not reprice each request by context length.
Extrapolated workload behavior is not a measurement.

## Retry a failed calibration

The command returns nonzero if a worker fails or expected output is absent. It stops its own process group,
including TP workers, and keeps partial files for diagnosis. Read the log named in the error. Fix model access,
unsupported structure or memory settings, then retry with a fresh output directory.

`--timeout` bounds each vLLM subprocess or SGLang startup, in seconds. It does not bound SGLang's request loop.
A stalled request is subject to the engine driver's request timeout. Ctrl-C stops the CLI-owned server.
Do not reuse costs across GPUs, TP sizes, model revisions, precision, attention backends or engine builds:
the server checks the tables' `key.json` (GPU, TP size, model pair, engine version, dtype) and refuses a
mismatch; the attention backend is not part of the key. Tables are also validated at load: a curve that is not
finite and positive with an increasing query index is refused with its directory, and the cost-model modes
refuse a target / draft cost ratio outside 5..100 or a verify cost at 8 tokens below 0.85x the cost at 1 token
(0.7x for MoE targets).

## Chain widths

vLLM's chain loop verifies a fixed number of drafts per batch size, so the width of a chain is chosen once per batch
size at startup (`eagle_harness/core/chain_width.py`). A round of width `w` gains `1 + a_1 + ... + a_{w-1}` tokens per
request, where `a_d` is the share of rounds that accept the first `d` drafted tokens, and takes `V(B, w) + (w - 1) D(B)`:
the target's verify forward of `w` tokens per request and `w - 1` draft steps at batch size `B`, from the cost tables
(median over the context intervals). The width with the most tokens per unit of time wins. Width `1` verifies
only the root and can still run draft KV extension. Width `0` means native in-engine decode with no draft
forward or draft KV update. It requires separate measured round times and adapter support.
On a vLLM host exposing `SpeculativeConfig.native_decode_batch_sizes`, automatic chains pass
selected width-0 ranges separately from the draft-count schedule. The host stops draft work,
drains pending verification and uses native target metadata and graphs. Suspension lasts until
request finish and spreads to requests merged into the same batch, because resuming would require
draft KV catch-up. Fresh requests can speculate after those requests finish.
This requires synchronous Model Runner V2 with prefix caching and KV transfer disabled, and eager
execution or full decode graphs. Trees and unsupported host configurations refuse native OFF.
Older hosts still reject a selected width 0. Width 1 and explicit K=0 schedules keep their
root-only speculative semantics. Draft weights and KV reservations stay allocated, so width-0
round times must be measured with the plugin installed rather than copied from a separate no-spec engine.
The separate startup comparison is in `tools/serving_choice.py`, outside the runtime policy.
Small batches read weights, so the widest chain wins there; large batches compute, and the width narrows.

The `a_d` come from `<data_path>/accept/chain.json`:

```json
{"drafts": 41234, "accepted_per_position": [36610, 32012, 27740]}
```

These are vLLM's `spec_decode_num_drafts` and `spec_decode_num_accepted_tokens_per_pos` counters after serving the
workload with the widest chain. Acceptance depends on the workload (short answers accept differently from long
ones). The policy reuses this profile across batch sizes, so measure representative requests and check that
it remains appropriate at the concurrency you serve. Widths past the measured positions are not
considered. Without the file, or without timings that price a chain at every calibrated batch size, chains verify
`max_verify_width` tokens at every batch size.

### Served round costs

Use served round costs when isolated forward timings do not predict throughput. Put the measured periods in
`<data_path>/rounds/chain.json`:

```json
{"rounds": [{"batch_size": 64, "width": 1, "round_us": 10000, "count": 120},
            {"batch_size": 64, "width": 4, "round_us": 22000, "count": 120}]}
```

Each period covers a whole decode round, including scheduling, sampling and drafting. These measurements replace
`V(B, w) + (w - 1) D(B)` in the chain policy. The acceptance profile is still required. Measure width 1 at every
batch size so the policy can choose no drafts when wider chains lose. Keep the engine settings and the maximum
speculative-state reservation identical across widths, warm the full workload, and exclude prefill and batch changes.

Width 1 disables proposals within the speculative engine. It does not unload the draft, release its KV cache or
remove the target's speculative-state reservation. Compare throughput against a separate no-spec engine at the
concurrency you serve. A zero-draft schedule does not establish the same cost as native no-speculation decoding.

The shared chain policy uses accepted-prefix counts and measured round costs to choose tokens per second,
not the largest acceptance length. Without a complete matching profile, do not assume the runtime has selected
an optimal width. The configuration reference describes [chain scheduling](configuration.md#engine-arguments).

## GEMM solutions (SGLang, ROCm)

The BLAS library's default heuristic can pick a slow kernel for the skinny GEMMs of a decode round.
PyTorch's TunableOp times solutions for each GEMM shape offline. EAGLE-4 serves the saved picks from
`<data_path>/gemm_solutions.csv` with GEMM tuning disabled, so server starts reuse the recorded choices.
Every rank uses the same saved kernels.

1. Record the serving shapes once: start the server with `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
   PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 PYTORCH_TUNABLEOP_UNTUNED_FILENAME=/tmp/untuned.csv` and run a short
   workload (`tools/greedy_parity.py run` at the serving concurrency). TunableOp writes `/tmp/untuned0.csv`
   (one file per rank).
2. `python tools/tune_gemm_solutions.py --data-path "$DATA" --untuned /tmp/untuned0.csv` tunes the decode-round
   shapes (at most `--max-rows` rows, default 64; prefill shapes keep the default solution) with inputs rotated
   through a 1 GB buffer so weights stream from HBM as they do in serving, and writes the file.
   For a larger serving batch, use `--decode-rows 1-8 4 8,16,32` instead of `--max-rows` to select only
   row counts for batches 1 through 8, draft top-k 4, and verify widths 8, 16, and 32. This avoids tuning
   unrelated prefill lengths from the same recording. It does not invent shapes absent from the recording.

The file carries the PyTorch, ROCm, hipBLASLt/rocBLAS and GPU identity it was tuned on; on any other stack the
runtime logs a warning and keeps the default kernels. Tuned picks are not bit-identical to the default ones
(another accumulation order), so re-run the greedy parity check after tuning.
