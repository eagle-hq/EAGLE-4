# Troubleshooting

Start with `eagle-harness-check` (also `python -m eagle_harness.compat`). It checks the installed engines against the
hooks the plugin needs, without starting a server, and exits 1 when something is wrong.

```bash
eagle-harness-check            # engines and hooks
eagle-harness-check --kernels  # also loads the native CUDA library on the visible GPU
```

## Reading `eagle-harness-check`

The first line names the eagle-harness version and where Python imports it from. Then one line per engine:

| line | meaning | what to do |
|---|---|---|
| `sglang <version>: ok, every hook matches` | the plugin registers in this engine | nothing |
| `... ok, every hook matches (untested version, expected sglang>=0.5.23.dev0,<0.6)` | the hooks match, but this engine version was not tested with this eagle-harness | it runs, test before you serve |
| `sglang <version>: FAILED` followed by `eagle-harness <v> is disabled: ... lacks N of the hooks it needs (...)` | the engine has no eagle-harness hooks, or a different version of them | install the engine build listed in [Compatibility](compatibility.md), or the eagle-harness release your engine build pins |
| `vllm: not installed` | no importable vLLM in this environment | nothing, if you serve with another engine |
| `no supported host installed (one of: sglang, vllm, tokenspeed)` | no engine imports | install one, in the same environment as eagle-harness |

With `--kernels`, the last line says where the native library came from:

| line | meaning |
|---|---|
| `native CUDA kernels: prebuilt: <path>` | the wheel's library for this CUDA major version loaded |
| `native CUDA kernels: from $EAGLE_HARNESS_NATIVE_LIB: <path>` | your own build loaded |
| `native CUDA kernels: built by nvcc on first use: <path>` | no prebuilt library matched, nvcc built one |
| `native CUDA kernels: not loaded (the Triton kernels run; see the warning above)` | no library could load. The results are the same, the candidate head is slower. The warning before it names each reason. |

On a machine without engines and without a GPU the output is:

```text
eagle-harness 0.1.1 (.../eagle_harness/__init__.py)
sglang: not installed
vllm: not installed
tokenspeed: not installed
no supported host installed (one of: sglang, vllm, tokenspeed)
```

## Server errors at startup

| message | cause | fix |
|---|---|---|
| `Unknown speculative algorithm name: EAGLE_HARNESS` (sglang), vLLM rejects the method `eagle_harness`, or TokenSpeed does not know the algorithm `EAGLE_HARNESS` | the plugin did not register | run `eagle-harness-check` and read the error the engine logged at startup |
| `Cost tables under ... were measured on another setup (stored vs this server): {...}` | `data_path` tables keyed for another GPU, TP size, model pair, engine build or dtype | calibrate for this setup, or drop `data_path` to measure the costs on the served rounds |
| `TokenSpeed serve mode (the default) needs data_path or a complete explicit chain schedule ...` | serve mode, set explicitly or by default, with no cost tables and no complete chain schedule (TokenSpeed) | set `data_path` to matching TokenSpeed costs ([Calibration](calibration.md)); static mode is not a correctness-qualified substitute |
| `config: unknown keys ['intensty']; valid keys are [...]` | a misspelt key | use a key from the list ([Configuration](configuration.md)) |
| `mode: 'fast' is not one of static, serve, eval, calibrate.` | a bad value | use one of the listed values |
| `EAGLE_HARNESS needs --speculative-num-steps, --speculative-eagle-topk and --speculative-num-draft-tokens ...` | sglang got part of the tree shape, or a chain config without one | pass all three, or none of them for the default tree (depth 12, top-k 4, verify width 16) |
| `max_running_requests=... exceeds the tree draft batch limit 512.` | the explicit request cap exceeds the supported range | set `--max-running-requests` to at most 512, and check that KV and recurrent-state memory can hold the requested batch |
| `eagle-harness kernels are off: ... is older than NVIDIA sm_80 (A100).` | the GPU is outside the kernel tiers | use a supported GPU ([Kernels](kernels.md)) |
| `page_size > 1 needs --speculative-draft-attention-backend flashinfer, fa3 or fa4 for EAGLE_HARNESS.` | the draft backend cannot serve paged KV | set one of those backends |
| `Unexpected FlashInfer page size 128 without trtllm-gen GQA` (vLLM) | the draft runs on FlashInfer, which serves a multi-head draft only on pages up to 64 tokens; a hybrid target needs 128 or more | set `"attention_backend": "FLASH_ATTN"` in `--speculative-config` (the draft's backend only) |
| `num_speculative_tokens=... disagrees with method_config.max_verify_width=...` (vLLM) | both were set and do not differ by one | set only `max_verify_width` |
| `the draft tree has fewer nodes than the verify width needs; raise tree_depth or node_top_k` (vLLM) | the tree cannot fill the verify width | raise `tree_depth` or `node_top_k`, or lower `max_verify_width` |
| `EAGLE_HARNESS on TokenSpeed supports static and serve modes` (TokenSpeed) | an unsupported evaluation or calibration mode | use `serve` with matching cost tables for the checked path; static remains unqualified |
| `tree drafting topk=... exceeds verify width ...` (TokenSpeed) | a draft step needs more write locations than the verify-sized buffer holds | raise `--speculative-num-draft-tokens` or lower top-k |
| `FlashInferSamplingBackend cannot verify draft trees; use --sampling-backend greedy or triton` (TokenSpeed) | TokenSpeed's default sampling backend | pass `--sampling-backend greedy` |
| `--speculative-num-draft-tokens N verifies more nodes than a tree of depth ... holds` (TokenSpeed) | the tree cannot fill the verify width | raise `--speculative-num-steps` or `--speculative-eagle-topk`, or lower the width |
| `EAGLE-4 drafts with [...] are not ported to eagle-harness yet (dense drafts only).` | a draft structure the engine does not build (see the FAQ, "Which models work?") | use a supported single-layer EAGLE-4 draft |

## Requests refused

sglang refuses requests with multimodal inputs (`Speculative algorithm EAGLE_HARNESS does not support multimodal
inputs.`). Serve them from a server without the method.

## Slower than expected

- Check that the cost tables match your GPU, TP size, tree shape and engine build ([Calibration](calibration.md)).
  A mismatched `key.json` is refused. Unkeyed legacy tables can load with a warning, and the key does not
  encode every setting, including the attention backend.
- Look for the startup warning `native CUDA kernels unavailable`. The Triton fallback gives the same results more
  slowly.
- On vLLM, `no cost tables price a draft step ...` means the lossless stop drafts every step that could
  still change the verify set. A loaded `rounds/tree.json` logs
  `eagle_harness: tree verify width per batch size from <path> (sha256 <digest>)` at startup; vLLM's log level
  applies to EAGLE-4's messages, so that line needs INFO.
- Run `python -m eagle_harness.kernels.arch` to see which kernel tier each GPU gets.
