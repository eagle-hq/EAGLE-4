---
type: tutorial
owner: eagle-harness
last_verified: 2026-10-21
---

# vLLM guide

Serve Qwen3.8-27B and its EAGLE-4 draft with vLLM. The first server uses a fixed tree and
needs no preparation. The measured recipe adds cost tables you calibrate once and a packaged schedule that
narrows the tree at high load.

## Start a server

Install the engine with the [vLLM install command](install.md#vllm-cuda), then start the server. The model and the
draft download from Hugging Face on the first start:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3.8-27B \
  --attention-backend CUSTOM --linear-backend flashinfer_cutedsl \
  --speculative-config '{"method": "eagle_harness", "model": "EAGLE-Corp/Qwen3.8-27B-EAGLE4", "method_config": {"mode": "static", "node_top_k": 4, "tree_depth": 14, "max_verify_width": 16}}'
```

Or run the published image, which has the engine and EAGLE-4 installed and starts `vllm serve` itself:

```bash
docker run --gpus all --ipc host -p 127.0.0.1:8000:8000 \
  -e HF_HOME=/hf -v ~/.cache/huggingface:/hf -e VLLM_USE_V2_MODEL_RUNNER=1 \
  ghcr.io/eagle-hq/eagle-4-vllm:0.1.3 \
  Qwen/Qwen3.8-27B \
  --attention-backend CUSTOM --linear-backend flashinfer_cutedsl \
  --speculative-config '{"method": "eagle_harness", "model": "EAGLE-Corp/Qwen3.8-27B-EAGLE4", "method_config": {"mode": "static", "node_top_k": 4, "tree_depth": 14, "max_verify_width": 16}}'
```

Both commands serve on one GPU. Add `--tensor-parallel-size 4` to serve on four, as the measured recipe below does.

The tree drafts 14 steps with 4 candidates per step and verifies 16 tokens. Tree verification needs vLLM's Model
Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`) and the `CUSTOM` attention backend, and an unquantized KV cache.

## Send a request

When the server logs `Application startup complete`, send a request from another terminal:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "Explain how a hash table works."}], "max_tokens": 256}'
```

The response holds the generated message. Send a few requests before you measure speed.

## The measured recipe

The TP4 vLLM rows of the [support matrix](support-matrix.md#011-speed-by-engine-and-concurrency) stop the tree early when another
draft step costs more than it gains, from cost tables measured on your GPUs. At 20 or more running requests the
packaged rounds file narrows the verified tokens from 16 to 4
([high-load rows](support-matrix.md#010-high-load)). Both live in one directory.

Stop any server on these GPUs, then calibrate once and add the rounds file:

```bash
eagle-harness calibrate --engine vllm --model Qwen/Qwen3.8-27B --draft EAGLE-Corp/Qwen3.8-27B-EAGLE4 --tp 4 \
  --engine-args '{"linear_backend":"flashinfer_cutedsl"}' --output calibration-vllm-tp4
mkdir -p calibration-vllm-tp4/rounds
python -c 'import shutil; from importlib.resources import files; shutil.copy(files("eagle_harness.core") / "profiles/vllm-qwen38-tp4/rounds/tree.json", "calibration-vllm-tp4/rounds/")'
```

Calibration refuses an existing output directory. It measures batch-one costs, not full high-concurrency serving,
and new measurements are not the exact table bytes of the release benchmark. The rounds file is measured for this
model pair on B300 at TP4 only ([profile notes](../eagle_harness/core/profiles/vllm-qwen38-tp4/README.md)).

Then serve with the measured settings, from the directory that holds `calibration-vllm-tp4`:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --max-num-seqs 64 \
  --seed 721 --no-enable-prefix-caching \
  --linear-backend flashinfer_cutedsl --reasoning-parser qwen3 --attention-backend CUSTOM \
  --speculative-config '{"method": "eagle_harness", "model": "EAGLE-Corp/Qwen3.8-27B-EAGLE4", "method_config": {"mode": "static", "node_top_k": 4, "tree_depth": 14, "max_verify_width": 16, "data_path": "calibration-vllm-tp4"}}'
```

With INFO logging the server prints `eagle_harness: tree verify width per batch size from <path>` when it loads the
rounds file. Without `rounds/tree.json` the verify width stays 16 at every batch size. Do not point `data_path` at
the packaged rounds-only directory, because it has no cost tables. Sampling, logprobs and structured output have
their own limits in the [known limits](known-limits.md#requests).
