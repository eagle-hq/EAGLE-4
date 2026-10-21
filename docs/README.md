---
type: explanation
owner: eagle-harness
last_verified: 2026-10-21
---

# EAGLE-4 documentation

To serve a model with EAGLE-4, install one engine and follow its guide. SGLang needs no preparation, vLLM
reaches its measured speed after a one-time calibration, and TokenSpeed runs on NVIDIA GPUs.

| Task | Guide |
|---|---|
| Install an engine with pip, or pick a Docker image | [Install](install.md) |
| Serve on SGLang | [SGLang guide](quickstart-sglang.md) |
| Serve on vLLM | [vLLM guide](quickstart-vllm.md) |
| Serve on TokenSpeed | [TokenSpeed guide](quickstart-tokenspeed.md) |
| Serve on AMD MI355X | [AMD guide](quickstart-amd.md) |
| See measured speed and correctness | [Support matrix](support-matrix.md) |
| Check unsupported or unmeasured behavior | [Known limits](known-limits.md) |
| Measure and fit serving costs | [Calibration](calibration.md) |
| Look up config keys and defaults | [Configuration](configuration.md) |
| Diagnose install, startup and generation errors | [Troubleshooting](troubleshooting.md) |
| Train, export and qualify a draft for another target | [New-model onboarding](new-model.md) |
| Understand plugin hooks and decode rounds | [Architecture](architecture.md) |
| Check kernel implementations and fallbacks | [Kernels](kernels.md) |
| Check host versions and release numbering | [Compatibility](compatibility.md) |
| Convert an existing EAGLE-4 configuration | [Migration](migration.md) |
| Find short answers | [FAQ](faq.md) |
