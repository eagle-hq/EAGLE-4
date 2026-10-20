---
type: how-to
owner: eagle-harness
last_verified: 2026-10-20
---

# Install eagle-harness 0.1.0

At release, install one patched engine per environment, then install `eagle-harness==0.1.0` from PyPI.
Stock engine packages do not provide the required plugin hooks. Choose the pip or Docker path below,
then follow that engine's quickstart.

## Release locations and availability

[Release settings](release.env) is the single source for package pins, wheel indexes and container tags.
The engine wheels belong to the GitHub release named by `EAGLE_RELEASE_ASSETS`. Its PEP 503 indexes
separate CUDA and ROCm wheels. The `eagle-harness` release path is PyPI, with its wheels and source distribution
also attached to the same GitHub Release. Publishing the public release runs `publish.yml` with PyPI trusted publishing.

These are release destinations, not evidence that publication has completed. The PyPI project and public
artifacts are not yet available for this draft. Fresh CUDA environments installed private wheels and served
real requests, and the `eagle-harness` wheel and source-distribution forms passed installation checks. Those checks
used a private index, not the public URLs. The private SGLang image also passed serving and correctness
checks. vLLM image qualification is pending, and the TokenSpeed image has not been built. Final release
artifacts and registry publication require separate checks.
If PyPI is unavailable after publication, download the matching `eagle-harness` wheel from the GitHub Release
named by `EAGLE_RELEASE_ASSETS`, verify its release checksum, and install it with
`python -m pip install /path/to/downloaded/eagle_harness-0.1.0-py3-none-any.whl` in the engine environment.
Before the release assets exist, neither path is available.
Do not replace an unavailable artifact with the stock engine or an unpinned package.

Use the release checkout containing these docs. Run commands from its root:

```bash
. ./docs/release.env
```

## Requirements

- Linux x86-64. Use Python 3.12 for CUDA engines and Python 3.10 for the ROCm wheel.
- NVIDIA wheels target CUDA 13.0 and require a compatible host driver. The serving recipes use B300 GPUs.
  Runtime compilation also needs a CUDA toolkit compatible with the host C/C++ headers.
  On the tested Ubuntu 26.04 / glibc 2.43 host, CUDA 13.0 `nvcc` fails with `rsqrt`/`rsqrtf`
  exception-specification errors. Use `export CUDA_HOME=/usr/local/cuda-13.2` for that host's JIT
  compilation, or use the release container's compatible toolchain. This does not change the wheel index.
- SGLang ROCm requires the measured ROCm 7.2 runtime on MI355X, including compatible PyTorch, Triton,
  AITER and FlyDSL. The measured PyTorch build is `2.9.1+rocm7.2.0.lw.git7e1940d4`.
  A CUDA PyTorch wheel is not a replacement. Use the complete ROCm runtime image rather than mixing
  arbitrary PyPI dependencies into it.
- Docker needs NVIDIA GPU access for CUDA, or `/dev/kfd` and `/dev/dri` access for ROCm.
- Obtain readable target and compatible draft weights before serving. The quickstarts use
  Qwen3.8-27B BF16 and its EAGLE-4 draft. Check the [support matrix](support-matrix.md) for other pairs.

## SGLang CUDA

Create a Python 3.12 environment and install the pinned CUDA engine:

```bash
python3.12 -m venv .venv-sglang
source .venv-sglang/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url "$EAGLE_CUDA_INDEX" \
  --extra-index-url "$EAGLE_TORCH_CUDA_INDEX" --extra-index-url "$EAGLE_NVIDIA_INDEX" \
  "sglang==$EAGLE_SGLANG_VERSION" 'flashinfer-python>=0.7.2'
python -m pip install "eagle-harness==$EAGLE_VERSION"
eagle-harness-check
```

The NVIDIA index supplies the pinned `cuda-tile==1.6.0rc5` wheel. It avoids relying on a source build
from PyPI. The private install check also succeeded with that exact dependency built from its PyPI source.

After the image is published and qualified, open it and check the hooks:

```bash
docker run --rm --gpus all --ipc host -it \
  -p 127.0.0.1:30000:30000 -v "$PWD/models:/models:ro" \
  --entrypoint bash "$EAGLE_SGLANG_IMAGE"
eagle-harness-check
```

Inside the container, use `/models` paths and `--host 0.0.0.0` in the
[SGLang CUDA quickstart](quickstart-sglang.md). The host port stays bound to loopback.

## SGLang ROCm on AMD

The private offline pip closure is qualified, but its fresh-install commands are deferred to the AMD
follow-up guide. No public pip or Docker installation command is qualified here. The ROCm wheel requires Python 3.10 (`cp310`), not Python 3.12,
and a complete compatible ROCm 7.2 runtime. A successful import or `eagle-harness-check` alone does not
qualify the dependency closure for serving.

The [AMD quickstart](quickstart-amd.md) records the measured serving recipe for an already-qualified
MI355X environment. Do not use that recipe to assemble a fresh runtime or install CUDA FlashInfer
packages into a ROCm environment.

## vLLM CUDA

Use a separate environment:

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url "$EAGLE_CUDA_INDEX" \
  --extra-index-url "$EAGLE_TORCH_CUDA_INDEX" "vllm==$EAGLE_VLLM_VERSION"
python -m pip install "eagle-harness==$EAGLE_VERSION"
eagle-harness-check
```

The vLLM image has been built, but its serving and correctness checks are pending. After qualification
and publication, mount a writable directory for the calibration output:

```bash
mkdir -p calibration
docker run --rm --gpus all --ipc host -it \
  -p 127.0.0.1:8000:8000 -v "$PWD/models:/models:ro" \
  -v "$PWD/calibration:/calibration" --entrypoint bash "$EAGLE_VLLM_IMAGE"
eagle-harness-check
```

Follow the [vLLM quickstart](quickstart-vllm.md). Inside the container, use `/models` paths,
place calibration output under `/calibration`, and bind the server to `0.0.0.0`.

## TokenSpeed CUDA, experimental

TokenSpeed requires the matching engine, scheduler and kernel wheels pinned in `release.env`.
Install all three from the CUDA release index. Stock scheduler and kernel wheels are not substitutes.
Tree-round CUDA sources compile at first use even though the runtime wheel has a `py3-none-any` tag.
The passing serve-mode check uses a host-compatible CUDA 13.2 JIT on Ubuntu 26.04 / glibc 2.43,
with `CUDA_HOME=/usr/local/cuda-13.2`; the original CUDA 13.0 JIT fails on that host. This does not
change the `cu130` wheel channel. The scheduler and kernel wheels contain native code:

```bash
python3.12 -m venv .venv-tokenspeed
source .venv-tokenspeed/bin/activate
python -m pip install --upgrade pip
: "${EAGLE_TOKENSPEED_AUX_REQUIREMENTS:?release native-wheel pins are required}"
# Intentional word splitting: this variable contains pinned package requirements.
python -m pip install --extra-index-url "$EAGLE_CUDA_INDEX" \
  --extra-index-url "$EAGLE_TORCH_CUDA_INDEX" \
  "tokenspeed==$EAGLE_TOKENSPEED_VERSION" $EAGLE_TOKENSPEED_AUX_REQUIREMENTS
python -m pip install "eagle-harness==$EAGLE_VERSION"
eagle-harness-check
```

No TokenSpeed image has been built or qualified. The following is the planned command after an
approved image is qualified and published, not an available installation path:

```bash
docker run --rm --gpus all --ipc host -it \
  -p 127.0.0.1:30000:30000 -v "$PWD/models:/models:ro" \
  --entrypoint bash "$EAGLE_TOKENSPEED_IMAGE"
eagle-harness-check
```

Follow the [TokenSpeed quickstart](quickstart-tokenspeed.md). Inside the container, use `/models` paths
and bind to `0.0.0.0`. This path is NVIDIA-only, not a ROCm recipe.

## Check the installation

`eagle-harness-check` must exit zero and report the installed engine's hook checks. A failure means the
engine does not provide the expected contract. Fix that before loading weights. This check does not
prove GPU kernels work or generation is correct. The quickstart's generated response is a separate check.

For the SGLang and vLLM calibration CLI, also run:

```bash
eagle-harness calibrate --help
```

Do not install multiple engines in one environment. Use only the trusted release indexes above.
The [troubleshooting guide](troubleshooting.md) covers hook failures and startup errors.

## Dockerfile sketches

The files under `docker/` document how to layer release wheels and this source checkout onto engine
runtimes. The private image checks used a prebuilt `eagle-harness` wheel instead of the sketches' source install,
so they do not qualify the Dockerfiles verbatim. The corrected SGLang TP4 image passed hook, dependency,
request and twenty-prompt correctness checks. vLLM image qualification is pending. TokenSpeed has no
built image, and ROCm has no container qualification. The ROCm and TokenSpeed sketches require an
explicit compatible `BASE_IMAGE`; do not infer a tested base from a successful syntax check.
Final rebuilds and publication under the release tags are separate release steps.
