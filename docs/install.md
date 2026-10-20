---
type: how-to
owner: eagle-harness
last_verified: 2026-10-20
---

# Install EAGLE-4 0.1.0

Use one patched engine per environment. CUDA paths install `eagle-harness==0.1.0` from PyPI.
The ROCm path uses a complete wheel bundle.
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
The private ROCm pip closure below was clean-installed and inference-tested. Its Docker recipe
has not been container-qualified. Publication remains a separate release step.
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
- SGLang ROCm requires Ubuntu 22.04, the ROCm 7.2 developer toolkit (`hipcc` and headers),
  a C++ compiler, Git, and an MI355X with a compatible host driver.
  Use the complete, hash-locked ROCm wheel bundle below. The upstream MI35x base and its
  custom PyTorch wheel use the CPython 3.10 ABI. Python 3.12 is not a supported substitute.
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

Use the complete ROCm bundle with Python 3.10, not an isolated engine wheel or the CUDA index.
The bundle contains `wheels/`, `requirements.lock`, `constraints.txt` and `SHA256SUMS`.
Until the complete bundle is published with the release assets, obtain it from the release owner.
The individual SGLang wheel in the engine index is insufficient.
[The lock](../docker/rocm/requirements.lock) pins every package and wheel hash. The
[constraints](../docker/rocm/constraints.txt) provide the corresponding exact versions.
Do not use `--no-deps`, a source-overlay `PYTHONPATH`, or `--system-site-packages`.

On the compatible ROCm host, set `EAGLE_ROCM_BUNDLE` to the absolute bundle directory:

```bash
: "${EAGLE_ROCM_BUNDLE:?set the absolute path to the complete ROCm wheel bundle}"
(cd "$EAGLE_ROCM_BUNDLE" && sha256sum -c SHA256SUMS)
unset PYTHONPATH
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3.10 -m venv .venv-sglang-rocm
source .venv-sglang-rocm/bin/activate
python -m pip install --no-index --find-links "$EAGLE_ROCM_BUNDLE/wheels" \
  --require-hashes -r "$EAGLE_ROCM_BUNDLE/requirements.lock"
python -m pip check
python -c 'import torch; assert torch.version.hip and torch.cuda.is_available(); print(torch.__version__, torch.version.hip)'
eagle-harness-check
```

The bundle pairs `torch==2.9.1+rocm7.2.0.lw.git7e1940d4` with
`triton==3.7.0+amd.rocm7.2.0.git89002410`, `apache-tvm-ffi==0.1.9`,
`xgrammar==0.2.1` and the HIP `sglang-kernel==0.4.7` wheel for `gfx950`.
The PyPI kernel wheel with the same version is a different, CUDA artifact.
The Torch wheel has build tag `1eagle`: only dependency metadata and wheel records differ
from the vendor wheel. Its runtime files are unchanged. The corrected metadata pins the Triton
version used by the upstream MI35x image, rather than retaining the vendor wheel's incompatible
Triton 3.5.1 pin. The lock's hash is required to select this build, not just its package version.

### Build the ROCm image

The Dockerfile uses the immutable `lmsysorg/sglang:v0.5.20-rocm720-mi35x` base and installs
this same closure in an isolated Python 3.10 environment. It does not reuse the base image's
editable SGLang installation. Container construction and serving must be qualified separately
before a release image is advertised as tested.

Build with Docker BuildKit and the complete local wheel bundle:

```bash
docker buildx build --platform linux/amd64 --load \
  --build-context "rocm-wheels=$EAGLE_ROCM_BUNDLE/wheels" \
  -f docker/sglang-rocm.Dockerfile -t eagle-4-sglang-rocm:0.1.0 .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc host -it -p 127.0.0.1:30000:30000 -v "$PWD/models:/models:ro" \
  --entrypoint bash eagle-4-sglang-rocm:0.1.0
python -m pip check
eagle-harness-check
```

Follow the [AMD quickstart](quickstart-amd.md). Inside a container, use `/models` paths and
`--host 0.0.0.0`. Do not install CUDA FlashInfer packages into this environment.

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

## Dockerfile build status

The files under `docker/` describe release image construction, not proof of tested image builds.
The private CUDA image checks used a prebuilt `eagle-harness` wheel instead of the sketches' source install,
so they do not qualify those Dockerfiles verbatim. The corrected SGLang TP4 image passed hook, dependency,
request and twenty-prompt correctness checks. vLLM image qualification is pending, TokenSpeed has no
built image, and ROCm has no container qualification.
The ROCm recipe pins its upstream base by digest and requires the complete wheel bundle as a named
build context. The TokenSpeed sketch requires an explicit compatible `BASE_IMAGE`.
Do not infer container qualification from a successful pip installation or syntax check.
Final rebuilds and publication under the release tags are separate release steps.
