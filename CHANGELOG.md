# Changelog

## [0.1.3]

### Features

* **pkg:** Support TokenSpeed through generic engine hooks, with EAGLE-4 policy and tree verification in the harness plugin. TokenSpeed serves EAGLE-4 in serve mode from calibrated cost tables. Static mode with the released EAGLE-4 draft is a known limit until 0.1.4.
* **pkg:** Check sampled output distributions through the TokenSpeed API.

### Performance

* **sglang:** Serve the quick-start command with independent request admission and native decoding at high load.
* **sglang:** Fill draft KV without prediction work for prompts admitted above the verify limit.

### Bug fixes

* **kernels:** Use 64-bit layer offsets when committing GDN records.
* **core:** Stop `stop_check.py` with a usage error when the workload file has fewer conversations than concurrency times waves, thanks to @kevin9327.

### Builds and installation

* **pkg:** Build engine wheels and release images from approved inputs.
* **pkg:** Explain server warm-up before throughput measurement.
* **pkg:** Document the TokenSpeed 128K-token configuration measured on 0.1.2.
* **pkg:** Document the TokenSpeed TVM helper's provenance.
* **pkg:** Install CUDA redistributables with `cuda_redist.py` on Python 3.10, thanks to @kevin9327.
* **pkg:** Keep added `License-File` headers in the metadata of wheels without a description, thanks to @kevin9327.
* **pkg:** Write the wheel index root page for an index with no artifacts yet, thanks to @kevin9327.
* **docs:** Fix grammar in the FAQ and known limits, thanks to @lunar-me.

## 0.1.2 (2026-10-23)


### Bug fixes

* **sglang:** decode natively wherever the served tree or chain would be slower than plain decoding, chosen from whole served intervals under the same load
* **sglang:** speculate without measuring native decode where speculation clearly wins, and check native decode's estimate only where the policy has one
* **sglang:** compile every KV-move program a tree round runs before serving, and keep rounds that stalled on one-time work or followed a switch out of the policy's prices
* **vllm:** `eagle-harness calibrate --engine vllm` completes: the calibration token budget fits one batch, tables are keyed by the engine's resolved models, and `serve.sh` loads them


### Performance

* **sglang:** start served trees and chains from packaged choices on the measured B300 setup
* **tokenspeed:** project the candidate head through the engine linear

## 0.1.1 (2026-10-21)


### ⚠ BREAKING CHANGES

* **core:** streamlined configuration: `"mode": "eval"`, `"ragged"`, `cost_model.ragged_*` and `draft_model.moe_gate_acc` are removed, and a config that sets them is refused at load with a pointer to the [migration guide](docs/migration.md).

### Features

* **sglang:** serve with only the draft path by default, on SGLang's default overlap scheduler
* **sglang:** install the SGLang CUDA engine `0.5.23.dev252+eagle.5`
* **tokenspeed:** install the TokenSpeed engine build measured for 0.1.1 (since withdrawn; use the current install guide)
* **tokenspeed:** negotiate the plugin contract by capabilities
* **vllm:** answer vLLM's suspend_drafting and verify_budget hooks
* **core:** fp32 arbiter for greedy-parity divergences above the near-tie gap; a candidate that is the fp32 argmax passes
* **pkg:** install ROCm from public indexes


### Bug fixes

* **sglang:** capture first-use graphs without flushing the allocator, so the first long-context requests start faster
* **sglang:** compile first-use kernels and the draft softmax at startup, and keep slow rounds out of live costs
* **core:** bound the effect of one slow round on a live-cost cell, and pool seed minima across context intervals
* **sglang:** size 16-token chain rounds with the chain-depth rule; narrower chains keep the tree rule
* **sglang:** record sampled proposals for recursive tree verification, and require complete sampled-distribution evidence
* **sglang:** accept the engine hooks' msgspec accept types and the relocated kv-indices kernel
* **kernels:** freeze the draft window at the last committed position
* **vllm:** measure served tree acceptance, apply measured tree rounds in serve mode, and choose the tree's narrowing batch size from round-weighted gains
* **vllm:** accept chain widths and tree rounds only when they were measured on the same setup
* **vllm:** serve without vLLM's varlen verify-graph flag
* **vllm:** show plugin messages in the server log through vLLM's logger
* **tokenspeed:** one serve default and one complete-schedule rule
* **pkg:** make sampling_check accept a no-spec server as its own reference, and load tools/sampling_check.py without requests
* **core:** import the parity HTTP client only when the fp32 arbiter reads from a parity run
* **pkg:** lock the published eagle-harness 0.1.0 wheel and the complete runtime for ROCm


### Performance

* **sglang:** sample temperature-1 trees from recorded proposals in one fused Philox draw
* **sglang:** use the engine's prefill CUDA graph in the release recipes

## 0.1.0 (2026-10-20)

- Serve EAGLE-4 drafts with SGLang on NVIDIA B300 and AMD MI355X.
- Serve EAGLE-4 drafts with vLLM and the experimental TokenSpeed integration.
- Use packaged cost profiles to select speculative decoding schedules.
- Install the `eagle-harness` package from wheels or source with the matching engine wheels and platform-specific quickstarts.

See the [0.1.0 release notes](docs/release-notes/0.1.0.md) for supported configurations and known limits.
