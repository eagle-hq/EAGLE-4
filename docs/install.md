---
type: how-to
owner: eagle-harness
last_verified: 2026-10-21
---

# Install EAGLE-4

To install EAGLE-4, create a Python 3.12 environment and run the one `pip install` command for your engine.
Each command installs `eagle-harness` from PyPI and the patched engine from the EAGLE-4 wheel index, because the
stock engine packages do not have the hooks EAGLE-4 needs. Use one environment per engine. If you prefer
containers, use a [published image](#docker-images) instead.

## Requirements

- Linux x86-64 and Python 3.12. On Ubuntu, `python3.12 -m venv` needs the `python3.12-venv` package.
- NVIDIA GPUs with a CUDA 13 driver. The measured recipes use B300 GPUs.
- For pip installs, a CUDA toolkit with `nvcc` and a C++ compiler for kernels compiled on the first start.
  The driver alone is not enough. On Ubuntu 26.04 / glibc 2.43, use CUDA 13.2
  ([toolkit setup](troubleshooting.md#install-problems)). Docker images include their build tools.
- For AMD MI355X, see [SGLang ROCm on AMD](#sglang-rocm-on-amd), which uses Python 3.10.

## SGLang CUDA

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "eagle-harness[sglang]" "sglang==0.5.23.dev252+eagle.5" \
  --extra-index-url https://eagle-hq.github.io/EAGLE-4/whl/cu130/ \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.nvidia.com
```

Then follow the [SGLang guide](quickstart-sglang.md).

## vLLM CUDA

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "eagle-harness[vllm]" "vllm==0.32.0rc1.dev283+eagle.4" \
  --extra-index-url https://eagle-hq.github.io/EAGLE-4/whl/cu130/ \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

Then follow the [vLLM guide](quickstart-vllm.md).

## TokenSpeed CUDA, experimental

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "eagle-harness[tokenspeed]" "tokenspeed==0.1.1+eagle.5" \
  "tokenspeed-scheduler==0.1.40+eagle.4" "tokenspeed-kernel==0.1.4+eagle.5" \
  --extra-index-url https://eagle-hq.github.io/EAGLE-4/whl/cu130/ \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

TokenSpeed runs on NVIDIA GPUs only. Then follow the [TokenSpeed guide](quickstart-tokenspeed.md).

## What the indexes and pins are for

| Index or pin | Why |
| --- | --- |
| `https://eagle-hq.github.io/EAGLE-4/whl/cu130/` | The patched SGLang, vLLM and TokenSpeed wheels. They carry a `+eagle.N` local version, so pip picks them only when the command pins that exact version. |
| `https://download.pytorch.org/whl/cu130` | PyTorch built for CUDA 13.0, which the engine wheels need. |
| `https://pypi.nvidia.com` | The `cuda-tile==1.6.0rc5` wheel that the SGLang engine pins. Without it pip builds that package from source. |
| `tokenspeed-scheduler`, `tokenspeed-kernel` | TokenSpeed's native packages, built from the same source as the engine. The stock wheels do not work with it. |

The engine wheels are assets of the [v0.1.0 GitHub release](https://github.com/eagle-hq/EAGLE-4/releases/tag/v0.1.0),
and the index pages link to them with SHA-256 hashes. Their sources are [eagle-hq/sglang](https://github.com/eagle-hq/sglang),
[eagle-hq/vllm](https://github.com/eagle-hq/vllm) and [eagle-hq/tokenspeed](https://github.com/eagle-hq/tokenspeed).
`eagle-harness` itself is published only on PyPI.
[docs/release.env](release.env) holds the same pins and URLs for scripts and image builds.
[docker/sglang/requirements.lock](../docker/sglang/requirements.lock) pins the SGLang CUDA runtime dependencies by
wheel hash for the SGLang image build.

## Docker images

Each image has one patched engine and `eagle-harness` 0.1.0 installed. The engine guides show the `docker run`
command for each one.

| Engine | Image |
| --- | --- |
| SGLang CUDA | `ghcr.io/eagle-hq/eagle-4-sglang:0.1.0` |
| vLLM CUDA | `ghcr.io/eagle-hq/eagle-4-vllm:0.1.0` |
| TokenSpeed CUDA, experimental | `ghcr.io/eagle-hq/eagle-4-tokenspeed:0.1.0` |
| SGLang ROCm | not published, [build it](#build-the-rocm-image) |

To run a newer `eagle-harness` in an image, run `pip install -U eagle-harness` inside the container.

## Check the installation

`eagle-harness-check` exits zero when the installed engine has every hook EAGLE-4 needs. It does not start a
server, so a served request is still the real test. The [troubleshooting guide](troubleshooting.md) explains
its output and the startup errors.

## SGLang ROCm on AMD

Install the complete ROCm closure with Python 3.10, not an isolated engine wheel or the CUDA index.
[The lock](../docker/rocm/requirements.lock) pins every package version and wheel hash, so pip installs
exactly the tested files and refuses anything else. The
[constraints](../docker/rocm/constraints.txt) list the same versions without hashes.
The wheels come from public indexes:

| Wheels | Index |
| --- | --- |
| `eagle-harness` and most dependencies | PyPI |
| The ROCm SGLang engine, plus the Torch, `sglang-kernel` and AITER builds described below | `EAGLE_ROCM_INDEX` |
| Triton `3.7.0+amd.rocm7.2.0.git89002410` | AMD's Triton index, `EAGLE_AMD_TRITON_INDEX` |
| Torchvision `0.24.0+rocm7.2.0.gitb919bd0c` | AMD's ROCm 7.2 wheel directory, `EAGLE_AMD_ROCM_WHEELS` |

The host needs Ubuntu 22.04, the ROCm 7.2 developer toolkit (`hipcc` and headers), a C++ compiler, Git, and an
MI355X with a compatible host driver. AMD's PyTorch build for ROCm 7.2 and the engine wheel use the CPython 3.10
ABI, so Python 3.12 is not a substitute. Do not use `--no-deps`, a source-overlay `PYTHONPATH`, or
`--system-site-packages`. Clone the repository for the lock file and the settings, then install from the
directory that contains the checkout:

```bash
git clone https://github.com/eagle-hq/EAGLE-4
. EAGLE-4/docs/release.env
unset PYTHONPATH
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3.10 -m venv .venv-sglang-rocm
source .venv-sglang-rocm/bin/activate
python -m pip install --require-hashes -r EAGLE-4/docker/rocm/requirements.lock \
  --extra-index-url "$EAGLE_ROCM_INDEX" --extra-index-url "$EAGLE_AMD_TRITON_INDEX" \
  --find-links "$EAGLE_AMD_ROCM_WHEELS"
python -m pip check
python -c 'import torch; assert torch.version.hip and torch.cuda.is_available(); print(torch.__version__, torch.version.hip)'
eagle-harness-check
```

The download is about 2.8 GB. The closure pairs `torch==2.9.1+rocm7.2.0.lw.git7e1940d4` with
`triton==3.7.0+amd.rocm7.2.0.git89002410`, `apache-tvm-ffi==0.1.9`, `xgrammar==0.2.1`, AITER and
the HIP `sglang-kernel==0.4.7` for `gfx950`. Three of these are not on their vendors' indexes in the
tested form, so the EAGLE-4 index serves them. Each is selected by its hash, not just its version:

- Torch, build tag `1eagle`, is AMD's ROCm 7.2 wheel with one dependency corrected. AMD's metadata pins
  Triton 3.5.1 and AITER uses Triton 3.6 or newer, so the rebuilt metadata pins the Triton build used
  by the upstream SGLang MI35x image instead. Only the metadata, `WHEEL` and `RECORD` files differ from
  AMD's wheel. Every runtime file and its license files are unchanged.
- `sglang-kernel`, build tag `1eagle`, is the HIP build of SGLang's kernel package for `gfx950`.
  PyPI's `sglang-kernel==0.4.7` is a different, CUDA build. The wheel adds SGLang's
  `THIRDPARTYNOTICES.txt` to its license files.
- AITER `0.1.21.dev48+g4ad998328.d20260918` is built from the AITER revision and patches that the
  upstream SGLang MI35x image uses. AMD publishes no wheel of that build.

[Third-party notices](../THIRD_PARTY_NOTICES.md#redistributed-rocm-wheels) lists their licenses and changes.
Then follow the [AMD guide](quickstart-amd.md). Do not install CUDA FlashInfer packages into this environment.

### Build the ROCm image

The Dockerfile uses the immutable `lmsysorg/sglang:v0.5.20-rocm720-mi35x` base and installs
this same closure in an isolated Python 3.10 environment. It does not reuse the base image's
editable SGLang installation. The ROCm image is being container-qualified and is not published yet
([known limits](known-limits.md#distribution)).

Download the locked wheels with the Python 3.10 environment above, then build with Docker BuildKit:

```bash
python -m pip download --require-hashes -r EAGLE-4/docker/rocm/requirements.lock \
  --extra-index-url "$EAGLE_ROCM_INDEX" --extra-index-url "$EAGLE_AMD_TRITON_INDEX" \
  --find-links "$EAGLE_AMD_ROCM_WHEELS" --only-binary :all: -d rocm-wheels
docker buildx build --platform linux/amd64 --load \
  --build-context "rocm-wheels=$PWD/rocm-wheels" \
  -f EAGLE-4/docker/sglang-rocm.Dockerfile -t eagle-4-sglang-rocm:0.1.0 EAGLE-4
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc host -it -p 127.0.0.1:30000:30000 -v "$PWD/models:/models:ro" \
  --entrypoint bash eagle-4-sglang-rocm:0.1.0
python -m pip check
eagle-harness-check
```

Inside a container, use `/models` paths and `--host 0.0.0.0`.
