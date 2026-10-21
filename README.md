<div align="center">

<img src="docs/assets/Eagle.svg" alt="Eagle" width="180">

<h1>EAGLE-4</h1>

<p>
  <b>One Flexible Drafter for All Batch Sizes and Hardware</b>
</p>

<p>
  <a href="EAGLE4_Paper.pdf"><b>Paper (EAGLE-4)</b></a> ·
  <a href="https://huggingface.co/EAGLE-corp/EAGLE4-Qwen3.8-27B"><b>EAGLE-4 draft for Qwen3.8-27B</b></a> ·
  <a href="docs/README.md"><b>Documentation</b></a>
</p>

<p align="center">
  <a href="https://github.com/eagle-hq/EAGLE-4/releases/tag/v0.1.1"><img src="https://img.shields.io/badge/eagle--harness-v0.1.1-orange.svg" alt="eagle-harness 0.1.1"></a>
  <a href="https://github.com/eagle-hq/EAGLE-4/issues" target="_blank"><img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance"></a>
  <a href="https://github.com/eagle-hq/EAGLE-4/pulls" target="_blank"><img src="https://img.shields.io/badge/Contributions-welcome-brightgreen.svg" alt="Contributions welcome"></a>
</p>

<p align="center">
  <a href="https://eagle.ai/"><img src="https://img.shields.io/badge/Website-eagle.ai-FCCC36?labelColor=171819&color=FCCC36" alt="Eagle website"></a>
  <a href="https://www.linkedin.com/company/eagle/posts/?feedView=all"><img src="https://img.shields.io/badge/LinkedIn-Eagle-FCCC36?logo=linkedin&labelColor=171819&color=2B2D30&logoColor=FCCC36" alt="Eagle on LinkedIn"></a>
  <a href="https://x.com/EagleCorp"><img src="https://img.shields.io/badge/X-@EagleCorp-FCCC36?logo=x&labelColor=171819&color=2B2D30&logoColor=FCCC36" alt="Eagle on X"></a>
  <a href="https://github.com/eagle-hq"><img src="https://img.shields.io/badge/GitHub-eagle--hq-FCCC36?logo=github&labelColor=171819&color=2B2D30&logoColor=FCCC36" alt="GitHub organization"></a>
  <a href="https://huggingface.co/EAGLE-Corp" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-EAGLE--Corp-FCCC36" alt="HuggingFace"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Eagle--4-FCCC36" alt="License"></a>
</p>

</div>

<br />

<p align="center">
  <img src="docs/assets/paper_results.png" alt="EAGLE-4 single-request speedup over vanilla decoding, from the paper" width="65%">
</p>

<p align="center">
  <sub>
    Figure 1(a) of the EAGLE-4 paper: Qwen3.8-27B on the SPEED-Bench qualitative split, 4× NVIDIA B300, one request, temperature 0.
    The numbers this release measured with EAGLE-4 are in the <a href="docs/support-matrix.md">support matrix</a>.
  </sub>
</p>

## EAGLE-4

EAGLE-4 (Extrapolation Algorithm for Greater Language-model Efficiency) is the latest generation of the EAGLE family
for accelerating large language model inference. One drafter adapts its speculative decoding strategy to the batch
size, the hardware and the workload, and keeps the target model's output distribution.

The EAGLE-4 draft uses a sliding-window draft layer, a low-rank candidate head and an append-only draft cache. Dense
and latent-MoE draft structures are implemented. Qwen3.8-27B with its
[EAGLE-4 draft](https://huggingface.co/EAGLE-corp/EAGLE4-Qwen3.8-27B) is the measured target pair.

## Engines

This repository is `eagle-harness`, a pip-installable package (`eagle_harness`) that serves EAGLE-4 drafts on
SGLang, vLLM and TokenSpeed through engine plugins. EAGLE-4 sizes the speculative tree from measured draft and
verification costs for the batch size, hardware and workload it is running on. It registers through hooks the engine
exposes and does not patch the engine at runtime.

The patched engines ship as pinned wheels and container images, not as source trees in this repository:

| Engine | Hardware | Wheel index | Image | Source |
| --- | --- | --- | --- | --- |
| SGLang | NVIDIA B300, CUDA 13.0 | [cu130](https://eagle-hq.github.io/EAGLE-4/whl/cu130/) | `ghcr.io/eagle-hq/eagle-4-sglang:0.1.0` | [eagle-hq/sglang](https://github.com/eagle-hq/sglang) |
| SGLang | AMD MI355X, ROCm 7.2 | [rocm](https://eagle-hq.github.io/EAGLE-4/whl/rocm/) | not yet published ([build it](docs/install.md#build-the-rocm-image)) | [eagle-hq/sglang](https://github.com/eagle-hq/sglang) |
| vLLM | NVIDIA B300, CUDA 13.0 | [cu130](https://eagle-hq.github.io/EAGLE-4/whl/cu130/) | `ghcr.io/eagle-hq/eagle-4-vllm:0.1.0` | [eagle-hq/vllm](https://github.com/eagle-hq/vllm) |
| TokenSpeed, experimental | NVIDIA CUDA only | [cu130](https://eagle-hq.github.io/EAGLE-4/whl/cu130/) | `ghcr.io/eagle-hq/eagle-4-tokenspeed:0.1.0` | [eagle-hq/tokenspeed](https://github.com/eagle-hq/tokenspeed) |

Stock engine packages do not carry the hooks. [Release settings](docs/release.env) is the one place that holds
every version pin, index URL and image tag. The wheels are also attached to the
[v0.1.0 GitHub release](https://github.com/eagle-hq/EAGLE-4/releases/tag/v0.1.0): the engines did not change in 0.1.1.
The 0.1.0 images carry `eagle-harness` 0.1.0.

## Measured results

The [support matrix](docs/support-matrix.md) reports every measurement with its conditions, separately for
loading, greedy correctness, request features and throughput. Throughput is generated tokens per second of
workload time, higher is better. The rows below are the shapes the quickstarts ship. Each one is the mean of
at least two server starts on the SPEED-Bench client at temperature 0.

| Engine | GPUs | Client concurrency | No-spec tokens/s | EAGLE-4 tokens/s | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| SGLang | 4× B300, TP4 | 1 | 216.3 | 735.7 | 3.40× |
| SGLang | 4× B300, TP4 | 8 | 1479.4 | 2820.6 | 1.91× |
| SGLang | 4× B300, TP4 | 64 | 3925.3 | 5281.3 | 1.35× |
| SGLang | 1× MI355X, TP1 | 8 | 552.7 | 1145.9 | 2.07× |
| SGLang | 1× MI355X, TP1 | 32 | 1269.2 | 1980.8 | 1.56× |
| vLLM | 4× B300, TP4 | 1 | 241.6 | 722.8 | 2.99× |
| vLLM | 4× B300, TP4 | 8 | 1699.9 | 2883.9 | 1.70× |

Each speedup divides EAGLE-4 throughput by the same engine's matched no-speculation control.
TokenSpeed is experimental; its measured concurrent throughput is in the
[support matrix](docs/support-matrix.md#tokenspeed-experimental). An unexplained greedy mismatch fails the
correctness gate even when the speculative server is faster.

## Install

Install one patched engine per environment, then `eagle-harness==0.1.1` from PyPI. The
[v0.1.0 GitHub release](https://github.com/eagle-hq/EAGLE-4/releases/tag/v0.1.0) carries the engine wheels, not `eagle-harness`.
Use Python 3.12 for the CUDA engines and Python 3.10 for the ROCm wheel. The commands read their pins from the
release settings file in the checkout. Run them, and start servers, from the directory that contains the checkout.
Inside it, `python -m` would import the source tree's `eagle_harness/` instead of the installed wheel.

```bash
git clone https://github.com/eagle-hq/EAGLE-4
. EAGLE-4/docs/release.env
```

SGLang on CUDA:

```bash
python3.12 -m venv .venv-sglang && source .venv-sglang/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url "$EAGLE_CUDA_INDEX" \
  --extra-index-url "$EAGLE_TORCH_CUDA_INDEX" "sglang==$EAGLE_SGLANG_VERSION" 'flashinfer-python>=0.7.2'
python -m pip install "eagle-harness==$EAGLE_HARNESS_VERSION"
eagle-harness-check
```

Or open the release image, which has the engine and EAGLE-4 installed. The 0.1.0 image carries `eagle-harness`
0.1.0; upgrade it inside the container:

```bash
docker run --rm --gpus all --ipc host -it \
  -p 127.0.0.1:30000:30000 -v "$PWD/models:/models:ro" \
  --entrypoint bash "$EAGLE_SGLANG_IMAGE"
python -m pip install -U eagle-harness==0.1.1
eagle-harness-check
```

`eagle-harness-check` exits zero when the installed engine provides every hook the plugin needs. It does not
prove that kernels run or that generation is correct, so send the quickstart request before you measure anything.

The [installation guide](docs/install.md) has the same steps for [vLLM](docs/install.md#vllm-cuda) and
[TokenSpeed](docs/install.md#tokenspeed-cuda-experimental), which needs its scheduler and kernel wheels from the
same index. SGLang on AMD MI355X installs a hash-locked closure from PyPI, the ROCm index and AMD's indexes:
follow [SGLang ROCm](docs/install.md#sglang-rocm-on-amd), then the [AMD recipe](docs/quickstart-amd.md).

## Serve

Serve Qwen3.8-27B BF16 with its EAGLE-4 draft on four B300 GPUs. SGLang prices the tree from costs it measures
during served rounds, so this recipe needs no offline calibration. The full command with its environment is in
the [SGLang quickstart](docs/quickstart-sglang.md):

```bash
python -m sglang.launch_server --model-path "$MODEL" --tp-size 4 --dtype bfloat16 \
  --trust-remote-code --attention-backend flashinfer --page-size 1 \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
  --disable-overlap-schedule --cuda-graph-backend-decode full \
  --mem-fraction-static 0.75 --context-length 196608 --max-total-tokens 196608 \
  --chunked-prefill-size 16384 --mamba-ssm-dtype bfloat16 --disable-radix-cache \
  --reasoning-parser qwen3 --random-seed 721 --skip-server-warmup \
  --host 127.0.0.1 --port 30000 \
  --speculative-algorithm EAGLE_HARNESS --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer \
  --speculative-attention-mode prefill --speculative-num-steps 12 \
  --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config '{"mode":"serve"}'
```

Then send a greedy request and check that the response carries generated text:

```bash
curl --fail http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain how a hash table works.","sampling_params":{"temperature":0,"max_new_tokens":128}}'
```

The first request compiles kernels and seeds the cost tables, so repeat it before timing.

The other engines have their own recipes. [vLLM](docs/quickstart-vllm.md) uses a static tree whose depth stops on
calibrated costs, produced once with `eagle-harness calibrate --engine vllm --model "$MODEL" --draft "$DRAFT" --tp 4`
while the GPUs are idle, plus the shipped rounds file. [SGLang on MI355X](docs/quickstart-amd.md) selects the
packaged MI355X cost profile automatically. [TokenSpeed](docs/quickstart-tokenspeed.md) is experimental and
CUDA-only, with a fixed-shape mode for checking generation without cost tables. The
[calibration guide](docs/calibration.md) explains the measurement, and [new-model onboarding](docs/new-model.md)
covers training and exporting a draft for another target.

## What it supports

- Drafts: EAGLE-4 drafts with the low-rank candidate head, tested with Qwen3.8-27B including its GDN layers.
  Mamba2 targets and latent-MoE drafts have adapters on SGLang and vLLM. Per-model correctness coverage is in the
  support matrix.
- Engines: SGLang (V2 speculative worker), vLLM (Model Runner V2) and experimental TokenSpeed, each through the
  eagle-harness hooks. On SGLang and vLLM a draft top-k of 1 gives a chain and a larger top-k gives a tree.
  TokenSpeed implements static and calibrated trees, chains and sampled trees, without prompt logprobs or native
  no-speculation rounds, and its coverage is limited to the configurations in the support matrix.
- GPUs: NVIDIA sm_80 and newer, and AMD CDNA2 to CDNA4, through Triton kernels. The kernel tests run on B300
  and the other architectures are compile-checked. MI355X is covered by the serving measurements in the support
  matrix. See [kernels](docs/kernels.md).
- Requests: text only. vLLM tree verification handles greedy and sampled requests and needs an unquantized KV
  cache. Constrained output takes the correctness fallback in [compatibility](docs/compatibility.md).
- Batch: SGLang and vLLM trees and chains admit up to 512 running requests within the engine's KV and
  recurrent-state capacity. The SGLang tree default is 32, so set `--max-running-requests` for larger batches.
  No pipeline parallelism. Admission is not a throughput or correctness guarantee. Check the
  [support matrix](docs/support-matrix.md) and the [known limits](docs/known-limits.md).

## Documentation

[Architecture](docs/architecture.md) · [Configuration](docs/configuration.md) · [Calibration](docs/calibration.md) ·
[Kernels](docs/kernels.md) · [Compatibility](docs/compatibility.md) · [Troubleshooting](docs/troubleshooting.md) ·
[FAQ](docs/faq.md) · [Release notes](docs/release-notes/0.1.0.md) · [Contributing](CONTRIBUTING.md) ·
[Security](SECURITY.md)

## Citation

If you use EAGLE-4, cite the EAGLE-4 paper. EAGLE-4 builds on EAGLE, EAGLE-2 and EAGLE-3, which
[`CITATION.cff`](CITATION.cff) also lists.

```bibtex
@misc{eagle4,
    author = {EAGLE-4 Team},
    title = {{EAGLE-4}: One Flexible Drafter for All Batch Sizes and Hardware},
    year = {2026},
    note = {Preprint}
}
```

## Acknowledgements

- Thanks to Aditya Nandakumar (@proaditya) and the contributors to the original EAGLE-4 PR 1 for the ROCm port of EAGLE-4 on AMD Instinct MI355X.
- Thanks to Andy Luo (@andyluo7) and the contributors to the original EAGLE-4 PR 2 for the AMD AITER and Triton kernels on GFX950.
- Thanks to Andy Luo (@andyluo7), Aditya Nandakumar (@proaditya) and the contributors to the merged AMD series in the original EAGLE-4 PR 4 for AITER/FA3 and Split-KV on GFX950.
- Thanks to Nicolas Perrin (@nperrin-fr) and the TokenSpeed/LightSeek contributors for the TokenSpeed implementation in the original EAGLE-4 PR 3, including the MLP L2 warm-ahead and TGV SwiGLU work used in the TokenSpeed fork.
- SPEED-Bench is by NVIDIA. Model weights are licensed by their owners.

## License and trademark policy

Eagle 4 is **source-available software, not OSI-approved open-source software**.
The terms of the [Eagle 4 Source-Available License](LICENSE) govern its use,
modification, and distribution, and incorporate the [Eagle Trademark Policy](TRADEMARKS.md).
See also the [NOTICE](NOTICE) file.

Commercial Use of a Competing Offering requires a separate written commercial
license regardless of revenue. Other Commercial Use is permitted only within
the conditions of the license, including the US $20 million Aggregate Revenue
threshold measured over any consecutive 12-month period. If the threshold is
exceeded, a separate commercial agreement is required before beginning or
continuing any Commercial Use. The license specifies that the requirement
continues after a subsequent revenue decrease.

Downloading, cloning, forking, copying, accessing, modifying, using, hosting,
distributing, or otherwise exercising any licensed right constitutes acceptance
of the Source-Available License and incorporated Eagle Trademark Policy.
Mere viewing or inspection does not itself constitute trademark use.

For separate commercial terms, see the
[Commercial License Agreement](COMMERCIAL-LICENSE-AGREEMENT.md) and contact
[licensing@eagle.ai](mailto:licensing@eagle.ai). Publishing that agreement does
**not** grant commercial-license rights; it becomes binding only as stated in
its Section 1. Files derived from
sglang, vLLM, FlashInfer and other projects keep those projects' notices, listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Contributing

By submitting, uploading, pushing, committing, proposing, or otherwise providing
a pull request, patch, issue attachment, model, weight, dataset, documentation,
or other contribution to this repository, you agree to the [Eagle Contribution
License and Submission Terms](docs/legal/Eagle-Contribution-Terms-v1.0.pdf)
(PDF). If you do not agree, do not submit a contribution. If you act for an
employer, client, partner or other entity, you represent that you have authority
to submit the contribution and have obtained all required permissions.
Submitting a contribution does not change the license of existing Eagle code;
Eagle AI Corp controls outbound licensing for accepted contributions. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the licensing structure and for how to
build, test and open a pull request.
