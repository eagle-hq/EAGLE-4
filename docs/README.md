---
type: explanation
owner: eagle-harness
last_verified: 2026-10-19
---

# eagle-harness documentation

Install a pinned engine and check generation with live costs on SGLang or calibrated costs and the shipped
batch-dependent tree schedule on vLLM.
TokenSpeed is experimental. Measure the setup you will deploy.
Start with the measured model pairs rather than treating config admission as proof of support.

| Task | Guide |
|---|---|
| Install wheels or run a release Docker image | [Install](install.md) |
| Generate text on SGLang CUDA | [SGLang quickstart](quickstart-sglang.md) |
| Generate text on AMD MI355X | [AMD quickstart](quickstart-amd.md) |
| Generate text on experimental TokenSpeed | [TokenSpeed quickstart](quickstart-tokenspeed.md) |
| Generate text on vLLM | [vLLM quickstart](quickstart-vllm.md) |
| Measure and fit serving costs | [Calibration](calibration.md) |
| Train, export and qualify a draft for another target | [New-model onboarding](new-model.md) |
| Check measured models, TP sizes and request features | [Support matrix](support-matrix.md) |
| Check unsupported or unmeasured behavior | [Known limits](known-limits.md) |
| Look up config keys and defaults | [Configuration](configuration.md) |
| Understand plugin hooks and decode rounds | [Architecture](architecture.md) |
| Check kernel implementations and fallbacks | [Kernels](kernels.md) |
| Check host versions and release numbering | [Compatibility](compatibility.md) |
| Diagnose startup and generation errors | [Troubleshooting](troubleshooting.md) |
| Find short answers | [FAQ](faq.md) |
