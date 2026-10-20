---
type: tutorial
owner: eagle-harness
last_verified: 2026-10-20
---

# TokenSpeed quickstart, experimental

Run the experimental TokenSpeed adapter on NVIDIA CUDA hardware. This is not a ROCm path.
The release serving recipe uses Qwen3.8-27B BF16 and its EAGLE-4 draft on four B300 GPUs.

## Before starting

Complete the [TokenSpeed installation](install.md#tokenspeed-cuda-experimental), including its scheduler
and kernel wheels, and run `eagle-harness-check`. Keep the CUDA 13.0 toolkit and `nvcc` available for
first-use tree-round compilation. Set `MODEL` and `DRAFT` to readable local directories.
The measured cost-model recipe also requires matching calibrated target and draft tables under
`COST_ROOT/cost/`. EAGLE-4 does not include a TokenSpeed calibration CLI or automatically select
an installed TokenSpeed profile. Do not substitute vLLM or ROCm tables.

If you do not have matching TokenSpeed costs, use the fixed-shape option below to check generation.
It does not reproduce the cost-model benchmark or establish the same performance.

## Start the server

With matching costs, construct the configuration using JSON encoding:

```bash
: "${COST_ROOT:?set the matching TokenSpeed TP4 calibrated cost directory}"
export COST_ROOT
STEPS=12
test -d "$COST_ROOT/cost"
CONFIG=$(python -c 'import json, os; print(json.dumps({"mode":"serve","data_path":os.environ["COST_ROOT"]}))')
```

Alternatively, select a fixed tree without cost files:

```bash
CONFIG='{"mode":"static"}'
STEPS=10
```

The static path supports at most ten draft steps because the engine stores its depth-first sort key
in 64 bits. Do not reuse the twelve-step cost-model bound with `mode=static`.

Use the release launch flags with the selected configuration:

```bash
: "${MODEL:?set the local Qwen3.8-27B BF16 target directory}"
: "${DRAFT:?set the local compatible EAGLE-4 draft directory}"
: "${CONFIG:?select cost-model or fixed-shape mode above}"
: "${STEPS:?set the step bound with the selected mode above}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
python -m tokenspeed.cli serve "$MODEL" --tensor-parallel-size 4 \
  --attention-backend trtllm --max-model-len 65536 --max-num-seqs 4 \
  --disable-kvstore --seed 721 --gpu-memory-utilization 0.6 --disable-prefix-caching \
  --host 127.0.0.1 --port 30000 --reasoning-parser qwen3 \
  --speculative-draft-model-path "$DRAFT" --speculative-num-steps "$STEPS" \
  --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm EAGLE_HARNESS --speculative-algorithm-config "$CONFIG" \
  --sampling-backend greedy
```

The running-request cap is four in the single-client release recipe. The concurrent release recipe
uses `--max-num-seqs 32` with the same TP4 and context settings. Neither setting promises a speedup
for your workload. The fixed-shape option changes the policy and draft-step bound, not the engine.
In Docker, bind the server to `0.0.0.0` inside the container and keep the published host port on loopback.
If using cost files in Docker, mount their directory read-only and set `COST_ROOT` to that container path.

## Generate text

In another terminal, read the served model identifier and send a greedy chat request:

```bash
MODEL=$(curl --fail --silent http://127.0.0.1:30000/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
export MODEL
python -c 'import json, os; print(json.dumps({"model":os.environ["MODEL"],"messages":[{"role":"user","content":"Explain how a hash table works."}],"temperature":0,"max_tokens":128}))' | \
  curl --fail http://127.0.0.1:30000/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @-
```

The response must contain a generated choice. A reasoning model may place its output in the reasoning
field rather than `message.content`. Repeat after warmup before timing.

## Experimental limits

Prompt logprobs, per-request sampled trees and native-OFF are not supported by this adapter.
This greedy quickstart does not enable sampled decoding.
See the [support matrix](support-matrix.md) for model-specific correctness and performance qualifications.
Stop the server with Ctrl-C before starting another engine on the same GPUs.
