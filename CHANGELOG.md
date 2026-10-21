# Changelog

## 0.1.1 (2026-10-21)


### ⚠ BREAKING CHANGES

* **core:** streamlined configuration: `"mode": "eval"`, `"ragged"`, `cost_model.ragged_*` and `draft_model.moe_gate_acc` are removed, and a config that sets them is refused at load with a pointer to the [migration guide](docs/migration.md).

### Features

* **sglang:** serve with only the draft path by default, on SGLang's default overlap scheduler
* **sglang:** install the SGLang CUDA engine `0.5.23.dev252+eagle.5`
* **tokenspeed:** install the TokenSpeed engine `0.1.1+eagle.5` (experimental)
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
