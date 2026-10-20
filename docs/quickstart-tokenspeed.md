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
and kernel wheels, and run `eagle-harness-check`. Tree-round sources need `nvcc` at first use.
On the tested Ubuntu 26.04 / glibc 2.43 host, the CUDA 13.0 JIT fails. The passing serve-mode
check uses a host-compatible CUDA 13.2 JIT with `CUDA_HOME=/usr/local/cuda-13.2`. Set that path
to the installed compatible toolkit before launching. The engine wheels still come from `cu130`.
Set `MODEL` and `DRAFT` to readable local directories.
The measured cost-model recipe also requires matching calibrated target and draft tables under
`COST_ROOT/cost/`. EAGLE-4 does not include a TokenSpeed calibration CLI or automatically select
an installed TokenSpeed profile. Do not substitute vLLM or ROCm tables.

Use `mode=serve` with matching costs for the checked path. Its twenty-prompt install check passed
with 4 exact and 16 fp32-tolerance results. Static mode is not a qualified substitute when cost files
are unavailable: its check has one fp32 gap of 0.178637, above the 0.125 tolerance.
These are first-divergence checks, not bitwise output equality or sampled-distribution qualification.

## Start the server

With matching costs, construct the configuration using JSON encoding:

```bash
: "${COST_ROOT:?set the matching TokenSpeed TP4 calibrated cost directory}"
export COST_ROOT
STEPS=12
test -d "$COST_ROOT/cost"
CONFIG=$(python -c 'import json, os; print(json.dumps({"mode":"serve","data_path":os.environ["COST_ROOT"]}))')
```

Use the release launch flags with the serve-mode configuration:

```bash
: "${MODEL:?set the local Qwen3.8-27B BF16 target directory}"
: "${DRAFT:?set the local compatible EAGLE-4 draft directory}"
: "${CONFIG:?construct the serve-mode configuration above}"
: "${STEPS:?set the serve-mode step bound above}"
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
for your workload. A TokenSpeed release image has not been built or qualified.
When a qualified image is available, bind the server to `0.0.0.0` inside the container and keep the published host port on loopback.
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
