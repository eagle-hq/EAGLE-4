---
type: reference
owner: eagle-harness
last_verified: 2026-10-20
---

# Compatibility and versioning

An engine must provide every hook the plugin uses. A version number alone does not prove compatibility.
Use the exact release pins in [release settings](release.env), then run the checks in
[installation](install.md). Hook ranges below describe admission, not a promise that stock upstream
packages include the hooks or that every configuration in the range has been measured.

## Engine and runtime selection

| Component | Hook admission range or requirement | Release setting or recipe |
|---|---|---|
| SGLang CUDA | `>=0.5.23.dev0,<0.6` with hooks | `EAGLE_SGLANG_VERSION` in release settings |
| SGLang ROCm | SGLang hooks plus the ROCm verification integration | `EAGLE_ROCM_VERSION`, [AMD recipe](quickstart-amd.md) |
| vLLM | `>=0.32.0rc1.dev283,!=0.32.0` with hooks and Model Runner V2 | `EAGLE_VLLM_VERSION` |
| TokenSpeed, experimental | `>=0.1.1,<0.2` with hooks and matching native packages | `EAGLE_TOKENSPEED_VERSION` and `EAGLE_TOKENSPEED_AUX_REQUIREMENTS` |
| FlashInfer, CUDA SGLang | `>=0.7.2` | CUDA installation, not ROCm |
| Python | Package metadata allows 3.10 and newer | Release recipes use 3.12 |
| PyTorch and GPU runtime | Match the engine's dependency closure | [CUDA and ROCm requirements](install.md#requirements) |
| GPU and model pair | Configuration-specific qualification | [Support matrix](support-matrix.md) |

The release index carries patched engines with local version labels. Install those exact versions rather
than assuming that the extras resolve to a compatible stock engine. TokenSpeed's scheduler and kernel
packages must come from the same release source as its runtime package.

The pinned vLLM build includes KV cache group sizing, a decode GEMM used with
`--linear-backend flashinfer_cutedsl`, and native target decoding for configured batch sizes.
Native target decoding stays off unless `native_decode_batch_sizes` is set.

The same ranges are in two places that a test keeps equal: the `sglang`, `vllm` and `tokenspeed` extras in
`pyproject.toml` and `HOSTS` in `eagle_harness/compat.py`.

## What happens at load time

Each engine process loads the plugin and checks the hooks:

| situation | result |
|---|---|
| every hook matches, version in range | the method registers |
| every hook matches, version outside the range | the method registers, and a warning says the pair is untested |
| a hook is missing or its parameters changed | nothing registers, one error names each mismatch and the fix, the engine runs without the method |
| hardware outside the kernel tiers | the server refuses `EAGLE_HARNESS` at startup with the reason |

The plugin never raises during loading, so installing eagle-harness cannot break an engine install. Run
`eagle-harness-check` to see the same verdict without starting a server ([Troubleshooting](troubleshooting.md)).

## Structured output on vLLM

Tree-mode batches containing JSON, regex, or forced-tool grammar requests emit only the root token of each verify
round. vLLM supplies grammar masks for a linear draft sequence, which are not valid for the different ancestry of a
tree node. EAGLE-4 rejects all non-root candidates in these batches rather than accept a token under the wrong
mask. This also disables multi-token acceptance for unconstrained requests sharing that batch. Chain mode continues
to use vLLM's own rejection sampler.

This is a correctness fallback, not accelerated tree-constrained decoding. Tree acceleration for these requests
requires the host to supply grammar masks for each node's actual ancestry. A token limit can still truncate a valid
JSON prefix; clients must check the finish reason and validate the completed response against their schema.

## Model-specific hooks

For Mamba2 / NemotronH targets, tree verification also requires `Mamba2SpecRows` and the Mamba2 metadata and mixer
integration of `LinearAttnVerifyExtension`. The model initialization refuses tree mode if that integration is absent.

Nemotron latent-MoE drafts require `SharedExperts.disable_aux_stream()`. The draft disables its shared expert's
auxiliary stream before graph capture because conditional graph bodies cannot join that stream. Other layers retain
their stream settings. These model-specific checks run when the model is constructed, after general plugin checks.

## Versioning

eagle-harness versions follow `MAJOR.MINOR.PATCH`. Releases are cut from Conventional Commits: before 1.0, a `feat`
commit or a breaking change raises the minor version and a `fix` or `perf` commit the patch version. Each release's
`CHANGELOG.md` entry lists them.

Pin the exact engine build in [release settings](release.env). The patched wheels use the release index,
not the upstream PyPI projects. Unpinned resolution can select a different build without the required hooks.
