---
type: how-to
owner: eagle-harness
last_verified: 2026-10-20
---

# Install EAGLE-4 0.1.1

Use one patched engine per environment. Every path installs from public package indexes:
`eagle-harness==0.1.1` from PyPI and the patched engine from the EAGLE-4 wheel index.
Stock engine packages do not provide the required plugin hooks. Choose the pip or Docker path below,
then follow that engine's quickstart.

## Release locations and availability

[Release settings](release.env) is the single source for package pins, wheel indexes and container tags.
The engine wheels belong to the GitHub release named by `EAGLE_RELEASE_ASSETS` (v0.1.0: the engines did not
change in 0.1.1). Its PEP 503 indexes
separate CUDA and ROCm wheels. The `eagle-harness` wheels and source distribution are published only to PyPI,
not to the GitHub Release. Publishing the public release runs `publish.yml` with PyPI trusted publishing.

Fresh CUDA environments installed the engine wheels and served real requests, and the `eagle-harness`
wheel and source-distribution forms passed installation checks. Those checks used a private index.
Fresh installs of the SGLang and TokenSpeed CUDA wheels from the public index, with `eagle-harness` from PyPI,
later served the measured results in the [support matrix](support-matrix.md). The published SGLang and vLLM
images passed serving and correctness checks, and the published TokenSpeed image passed a GPU check in serve
mode. The ROCm pip path below was clean-installed from the
public indexes and inference-tested on an MI355X. No ROCm image is published, and its Docker recipe has not been
container-qualified.
Do not replace an unavailable artifact with the stock engine or an unpinned package.

Clone the release checkout containing these docs and read its settings:

```bash
git clone https://github.com/eagle-hq/EAGLE-4
. EAGLE-4/docs/release.env
```

Run every command in this guide and in the quickstarts from the directory that contains the checkout,
not from inside it. `python -m` puts the current directory first on the import path, so a server started
inside the checkout imports the source tree's `eagle_harness/` instead of the installed wheel.
`eagle-harness-check` cannot detect that, because a console script does not import from the current directory.

## Requirements

- Linux x86-64. Use Python 3.12 for CUDA engines and Python 3.10 for the ROCm wheel.
- NVIDIA wheels target CUDA 13.0 and require a compatible host driver. The serving recipes use B300 GPUs.
  Runtime compilation also needs a CUDA toolkit compatible with the host C/C++ headers.
  On the tested Ubuntu 26.04 / glibc 2.43 host, CUDA 13.0 `nvcc` fails with `rsqrt`/`rsqrtf`
  exception-specification errors. Use `export CUDA_HOME=/usr/local/cuda-13.2` for that host's JIT
  compilation, or use the release container's compatible toolchain. This does not change the wheel index.
- SGLang ROCm requires Ubuntu 22.04, the ROCm 7.2 developer toolkit (`hipcc` and headers),
  a C++ compiler, Git, and an MI355X with a compatible host driver.
  Install the hash-locked ROCm closure below. AMD's PyTorch build for ROCm 7.2 and the
  engine wheel use the CPython 3.10 ABI. Python 3.12 is not a supported substitute.
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
python -m pip install "eagle-harness==$EAGLE_HARNESS_VERSION"
eagle-harness-check
```

The NVIDIA index supplies the pinned `cuda-tile==1.6.0rc5` wheel. It avoids relying on a source build
from PyPI. The private install check also succeeded with that exact dependency built from its PyPI source.

Or open the published image and check the hooks. The 0.1.0 images carry `eagle-harness` 0.1.0; upgrade it
inside the container:

```bash
docker run --rm --gpus all --ipc host -it \
  -p 127.0.0.1:30000:30000 -v "$PWD/models:/models:ro" \
  --entrypoint bash "$EAGLE_SGLANG_IMAGE"
python -m pip install -U eagle-harness==0.1.1
eagle-harness-check
```

Inside the container, use `/models` paths and `--host 0.0.0.0` in the
[SGLang CUDA quickstart](quickstart-sglang.md). The host port stays bound to loopback.

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

Do not use `--no-deps`, a source-overlay `PYTHONPATH`, or `--system-site-packages`.
On the ROCm host, from the directory that contains the checkout:

```bash
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
  Triton 3.5.1, while AITER needs Triton 3.6 or newer. The rebuilt metadata pins the Triton build used
  by the upstream SGLang MI35x image instead. Only the metadata, `WHEEL` and `RECORD` files differ from
  AMD's wheel. Every runtime file and its license files are unchanged.
- `sglang-kernel`, build tag `1eagle`, is the HIP build of SGLang's kernel package for `gfx950`.
  PyPI's `sglang-kernel==0.4.7` is a different, CUDA build. The wheel adds SGLang's
  `THIRDPARTYNOTICES.txt` to its license files.
- AITER `0.1.21.dev48+g4ad998328.d20260918` is built from the AITER revision and patches that the
  upstream SGLang MI35x image uses. AMD publishes no wheel of that build.

[Third-party notices](../THIRD_PARTY_NOTICES.md#redistributed-rocm-wheels) lists their licenses and changes.

### Build the ROCm image

The Dockerfile uses the immutable `lmsysorg/sglang:v0.5.20-rocm720-mi35x` base and installs
this same closure in an isolated Python 3.10 environment. It does not reuse the base image's
editable SGLang installation. Container construction and serving must be qualified separately
before a release image is advertised as tested.

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
python -m pip install "eagle-harness==$EAGLE_HARNESS_VERSION"
eagle-harness-check
```

Or use the published vLLM image, which passed serving and correctness checks. It carries `eagle-harness` 0.1.0;
upgrade it inside the container. Mount a writable directory for the calibration output:

```bash
mkdir -p calibration
docker run --rm --gpus all --ipc host -it \
  -p 127.0.0.1:8000:8000 -v "$PWD/models:/models:ro" \
  -v "$PWD/calibration:/calibration" --entrypoint bash "$EAGLE_VLLM_IMAGE"
python -m pip install -U eagle-harness==0.1.1
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
python -m pip install "eagle-harness==$EAGLE_HARNESS_VERSION"
eagle-harness-check
```

Or use the published TokenSpeed image, which passed a GPU check in serve mode. It carries `eagle-harness` 0.1.0;
upgrade it inside the container:

```bash
docker run --rm --gpus all --ipc host -it \
  -p 127.0.0.1:30000:30000 -v "$PWD/models:/models:ro" \
  --entrypoint bash "$EAGLE_TOKENSPEED_IMAGE"
python -m pip install -U eagle-harness==0.1.1
eagle-harness-check
```

Follow the [TokenSpeed quickstart](quickstart-tokenspeed.md). Inside the container, use `/models` paths
and bind to `0.0.0.0`. This path is NVIDIA-only, not a ROCm recipe.

## Check the installation

`eagle-harness-check` must exit zero and report the installed engine's hook checks. A failure means the
engine does not provide the expected contract. Fix that before loading weights. This check does not
prove GPU kernels work or generation is correct. The quickstart's generated response is a separate check.

From the directory where you start servers, confirm that Python imports the installed package:

```bash
python -c 'import eagle_harness; print(eagle_harness.__file__)'
```

The printed path must be inside the environment's `site-packages`. A path inside the checkout means
the server would run the source tree instead of the installed wheel.

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
request and twenty-prompt correctness checks. The published vLLM image passed the same checks, the TokenSpeed
image passed a GPU check in serve mode, and ROCm has no published image or container qualification.
The ROCm recipe pins its upstream base by digest and takes the downloaded locked wheels as a named
build context. The TokenSpeed sketch requires an explicit compatible `BASE_IMAGE`.
Do not infer container qualification from a successful pip installation or syntax check.
Final rebuilds and publication under the release tags are separate release steps.
