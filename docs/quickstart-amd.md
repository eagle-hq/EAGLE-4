---
type: tutorial
owner: eagle-harness
last_verified: 2026-10-20
---

# SGLang on AMD MI355X

Serve Qwen3.8-27B BF16 with its EAGLE-4 draft on one MI355X using the ROCm SGLang engine.
The matching MI355X cost profile is packaged with EAGLE-4 and selected automatically.
No `data_path` or profile environment variable is needed.

## Before starting

Complete the [ROCm installation](install.md#sglang-rocm-on-amd) in a fresh Python 3.10 environment,
then run `python -m pip check` and `eagle-harness-check`. Install the entire hash-locked closure, including
its corrected Torch wheel. An individually installed SGLang wheel is not a complete ROCm runtime.
Run the commands below from the directory that contains the EAGLE-4 checkout, not from inside it.
Inside the checkout, `python -m sglang.launch_server` imports the source tree's `eagle_harness/`
instead of the installed wheel.
Use the ROCm 7.2 runtime and locally downloaded target and draft directories. Profile selection compares
both local `config.json` files with the measured pair, so an unresolved Hub identifier is not sufficient.
The profile applies to MI355X, including VF, BF16, TP1 and no quantization. It is not a generic AMD profile.
Only enable `--trust-remote-code` for model code you have reviewed.

## Set the ROCm environment

Use the environment from the packaged-profile serving recipe:

```bash
: "${MODEL:?set the local Qwen3.8-27B BF16 target directory}"
: "${DRAFT:?set the local compatible EAGLE-4 draft directory}"
unset PYTHONPATH AMD_COST_PROFILE
export OMP_NUM_THREADS=4 AITER_USE_SYSTEM_TRITON=1
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_SET_CPU_AFFINITY=1 SGLANG_INT4_WEIGHT=0 SGLANG_USE_ROCM700A=1
export SGLANG_ROCM_DISABLE_LINEARQUANT=0 SGLANG_USE_AITER=1 SGLANG_MOE_PADDING=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_SPLITKV_VERIFY_FLYDSL=0 SGLANG_SPLITKV_VERIFY_FLYDSL_COMBINE=0
```

Do not copy source-overlay `PYTHONPATH` settings into the installed environment.
Keep HIP prelaunch disabled and rounded recurrence enabled, as in the packaged-profile measurements.
The environment switch named `ROCM700A` is retained by the ROCm engine despite the runtime's 7.2 version.

## Start the server

The measured eight-request recipe uses AITER target attention, Triton draft attention and explicit
HIP graph batch sizes. Its padding setting differs from the CUDA recipe:

```bash
python -m sglang.launch_server --model-path "$MODEL" --tp-size 1 --dtype bfloat16 \
  --host 127.0.0.1 --port 30000 --trust-remote-code --attention-backend aiter --page-size 1 \
  --disable-overlap-schedule --disable-cuda-graph-padding --disable-prefill-cuda-graph \
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 \
  --max-running-requests 8 --max-mamba-cache-size 8 --mem-fraction-static 0.90 \
  --context-length 32768 --chunked-prefill-size 8192 --max-total-tokens 65536 \
  --mamba-ssm-dtype bfloat16 --disable-radix-cache --reasoning-parser qwen3 \
  --random-seed 721 --skip-server-warmup \
  --speculative-algorithm EAGLE_HARNESS --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-model-quantization unquant --speculative-draft-attention-backend triton \
  --speculative-attention-mode prefill --speculative-num-steps 12 \
  --speculative-eagle-topk 4 --speculative-num-draft-tokens 32 \
  --speculative-algorithm-config '{"mode":"serve"}'
```

Look for `selected packaged MI355X cost profile` in the startup log. The profile path on that line
must be inside the environment's `site-packages`. A path inside the checkout means the server started
from inside it, so stop the server and start it from the parent directory. If the line is absent, check the
GPU, dtype, TP size and local model configurations before assuming the measured profile is in use.
An explicit `data_path` overrides automatic selection. The packaged profile includes measured costs
for batch sizes 1, 8 and 32, but that does not qualify every batch size or context length.
See the [support matrix](support-matrix.md) for correctness coverage.

## Generate text

In another terminal, wait for the model listing, then generate:

```bash
curl --fail http://127.0.0.1:30000/v1/models
curl --fail http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain how a hash table works.","sampling_params":{"temperature":0,"max_new_tokens":128}}'
```

The response must contain generated text and token metadata. The first request includes cold work because
the measured launch skips server warmup. Repeat it before timing. In Docker, bind the server to `0.0.0.0`
inside the container while publishing only the loopback host port.
Stop the server with Ctrl-C before changing its graph sizes or starting another GPU process.

## Installed-wheel qualification

The hash-locked Python 3.10 pip closure passed two fresh starts on one MI355X:

| Concurrency | No-spec tok/s | EAGLE-4 start 1 / start 2 tok/s | Speedup over no-spec |
| --- | ---: | ---: | ---: |
| 1 | 82.93 | 290.67 / 291.22 | 3.51× / 3.51× |
| 8 | 552.39 | 1151.43 / 1184.13 | 2.08× / 2.14× |

The no-spec control is our own installed engine on the same MI355X, target weights and workload,
with speculation disabled and the ordinary overlapping scheduler. It was measured once per
concurrency with twice the Mamba cache cap. Measurements use 44 conversations / 60 turns,
1,024 maximum new tokens, greedy decoding and a separate warmup pass. At concurrency 1,
the running-request cap, decode graph batch sizes and Mamba cache cap are all reduced to 1.
The packaged MI355X profile was selected without a source overlay or profile override.

Each start also passed 20 greedy prompts against an installed no-spec control. Both C1 starts
had 7 exact / 13 near-tie cases. C8 had 3 exact / 16 near-tie / 1 fp32-arbitrated case on start 1,
and 6 exact / 13 near-tie / 1 fp32-arbitrated case on start 2; no cases failed.
The gate checks the first divergence: native log-probability gap at most 0.13, otherwise an
fp32 forward on the exact common prefix must differ by at most two BF16 ULPs or select our token
as its argmax. This is tolerance-based parity, not identical output streams.
Docker construction and GPU-container qualification remain separate, uncompleted gates.
