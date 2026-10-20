---
type: tutorial
owner: eagle-harness
last_verified: 2026-10-20
---

# vLLM quickstart

Serve Qwen3.8-27B BF16 with its EAGLE-4 draft on four B300 GPUs using the release tree configuration.
Load the shipped rounds file into matching calibrated costs to enable its batch-dependent verification width.
The rounds file is packaged, but the target and draft `cost/` tables must be measured separately.

## Before starting

Complete the [vLLM installation](install.md#vllm-cuda) and run `eagle-harness-check`.
Set `MODEL` and `DRAFT` to readable local directories. This recipe is specific to the Qwen3.8-27B / EAGLE-4
v3 pair at TP4. Do not reuse its rounds file for another GPU, tensor-parallel size or model pair.
See the [support matrix](support-matrix.md) for other configurations.

## Prepare the costs and shipped rounds

If you already have matching calibrated TP4 costs, set `COST_ROOT` to that directory and skip calibration.
Otherwise, with the GPUs idle, create them with the installed CLI:

```bash
: "${MODEL:?set the local Qwen3.8-27B BF16 target directory}"
: "${DRAFT:?set the local compatible EAGLE-4 draft directory}"
export MODEL DRAFT
export COST_ROOT="$PWD/calibration-vllm-tp4"
eagle-harness calibrate --engine vllm --model "$MODEL" --draft "$DRAFT" --tp 4 \
  --engine-args '{"linear_backend":"flashinfer_cutedsl"}' --output "$COST_ROOT"
```

Calibration refuses an existing output directory. It measures batch-one costs, not full high-concurrency
serving. New measurements are not the exact table bytes used by the release benchmark. In Docker, choose
a fresh directory beneath the writable `/calibration` mount instead of `$PWD`.

Copy the shipped measured rounds without replacing the calibrated `cost/` tables:

```bash
: "${COST_ROOT:?set the matching calibrated TP4 cost directory}"
export COST_ROOT
TREE_ROUNDS=$(python -c 'from importlib.resources import files; print(files("eagle_harness.core").joinpath("profiles/vllm-qwen38-tp4/rounds/tree.json"))')
test -d "$COST_ROOT/cost"
mkdir -p "$COST_ROOT/rounds"
cp "$TREE_ROUNDS" "$COST_ROOT/rounds/tree.json"
```

This file keeps width 16 below batch size 20 and selects width 4 at larger batches. It does not enable
native-OFF. The [profile notes](../eagle_harness/core/profiles/vllm-qwen38-tp4/README.md) describe its scope.

## Start the server

Build the speculative configuration with JSON encoding so paths with spaces or quotes remain valid:

```bash
: "${MODEL:?set the target directory}"
: "${DRAFT:?set the draft directory}"
: "${COST_ROOT:?set the prepared cost directory}"
export MODEL DRAFT COST_ROOT
SPEC=$(python -c 'import json, os; print(json.dumps({"method":"eagle_harness","model":os.environ["DRAFT"],"method_config":{"mode":"static","node_top_k":4,"tree_depth":14,"max_verify_width":16,"data_path":os.environ["COST_ROOT"]}}))')
export VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
python -m vllm.entrypoints.cli.main serve "$MODEL" --tensor-parallel-size 4 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --max-num-seqs 64 \
  --seed 721 --host 127.0.0.1 --port 8000 --no-enable-prefix-caching \
  --linear-backend flashinfer_cutedsl --reasoning-parser qwen3 \
  --attention-backend CUSTOM --speculative-config "$SPEC"
```

Model Runner V2 and `CUSTOM` attention are required for this tree. The static shape supplies the bounds,
the calibrated costs supply depth stopping, and the rounds file selects width by batch size.
Without `rounds/tree.json`, the configured width stays fixed. Do not point `data_path` at the packaged
rounds-only directory because it lacks the depth-stop cost tables.

## Generate text

In another terminal, read the served model identifier and send a completion:

```bash
MODEL=$(curl --fail --silent http://127.0.0.1:8000/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
export MODEL
python -c 'import json, os; print(json.dumps({"model":os.environ["MODEL"],"prompt":"Explain how a hash table works.","temperature":0,"max_tokens":128}))' | \
  curl --fail http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' --data-binary @-
```

The response must contain a generated choice. Repeat after warmup before timing. In Docker, bind the server
to `0.0.0.0` inside the container while publishing the host port on loopback.
Sampling and logprobs have separate checks in the support matrix. This plain-text example does not qualify
multimodal requests or quantized KV caches. Stop the server with Ctrl-C before another calibration.
