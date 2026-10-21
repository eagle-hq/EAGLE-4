# Contributing to eagle-harness

Open an issue before you start a large change (a new engine hook, a new kernel, a change to the cost model or the
configuration keys), so the design can be agreed before the code. Small fixes can go straight to a pull request.

## Contribution terms

By submitting, uploading, pushing, committing, proposing, or otherwise providing a pull request, patch, issue
attachment, model, weight, dataset, documentation, or other contribution to this repository, you agree to the
[Eagle Contribution License and Submission Terms](docs/legal/Eagle-Contribution-Terms-v1.0.pdf) (PDF, v1.0,
7 October 2026). If you do not agree, do not submit a contribution. If you act for an employer, client, partner or
other entity, you represent that you have authority to submit the contribution and have obtained all required
permissions.

No signature, certification form, separate submission or other acceptance procedure is required, and there is no CLA
to sign and no sign-off to add to your commits. Merely viewing this repository or the terms does not accept them.
Contact [licensing@eagle.ai](mailto:licensing@eagle.ai) with questions about licensing, authority or permissions.

### Licensing structure

Submitting a contribution does **not** change the license of existing Eagle code.

Under the contribution terms, Eagle AI Corp controls outbound licensing for accepted contributions. Depending on the
project and use case, Eagle may distribute or use a contribution under an Eagle source-available license, an Eagle
commercial license, a proprietary license, Apache-2.0, a model, weight or dataset license, another open-source or
source-available license, or a future licensing structure the contribution terms permit. Do not assume that
submitting code here makes that code Apache-2.0, open source, or eligible for contribution to another project.

EAGLE-4 itself is source-available, not open source: the [Eagle 4 Source-Available License](LICENSE) governs the code
and incorporates the [Eagle Trademark Policy](TRADEMARKS.md). See [`NOTICE`](NOTICE) for the short-form notice.

## Set up

```bash
git clone https://github.com/eagle-hq/EAGLE-4 && cd EAGLE-4
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

Install SGLang, vLLM or TokenSpeed with the eagle-harness hooks ([Compatibility](docs/compatibility.md)) to run that engine's
tests. Each engine's tests skip without it.

## Test

```bash
pytest tests                      # CPU tests (the *_gpu.py files skip without a GPU)
pytest tests/test_*_gpu.py        # GPU kernel tests against their torch references, on a supported GPU
pipx run ruff==0.15.1 check .     # lint (line length 120)
```

A change to a kernel needs a test against a reference (torch or the kernel it replaces), and a performance claim
needs a measurement with its hardware, model, workload and baseline. `tools/greedy_parity.py` compares a speculative
server token for token with a server without speculation. `tools/bench_kernels.py` times kernels.

To build the release files on Linux, install the build tools and set the source and output paths.
The example assumes CUDA 12.8 and 13.0 toolkits are installed at the paths shown:

```bash
python -m pip install build auditwheel patchelf
export SOURCE_TREE="$PWD" OUT_DIR="$PWD/dist-release"
tools/build_wheel.sh "$SOURCE_TREE" "$OUT_DIR" /usr/local/cuda-12.8 /usr/local/cuda-13.0
python tools/check_dist.py "$OUT_DIR"
```

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, in the
imperative, with a body that says why and gives the evidence.

| part | values |
|---|---|
| type | `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `chore` |
| scope | `sglang`, `vllm`, `tokenspeed`, `kernels`, `core`, `pkg` |

Releases and the changelog are built from these messages: `feat` raises the minor version, `fix` and `perf` the
patch version. Mark a breaking change with `!` after the scope or a `BREAKING CHANGE:` footer. Keep one logical
change per commit.

## Pull requests

The pull request template asks for a summary, the tests you ran, third-party material and AI-assisted content. Fill
in every section, and write "None" where one does not apply.

## Third-party code

Say in the pull request where any copied or adapted code comes from and under which license. A file derived from
another project keeps that project's copyright notices and names its source in a header, and the project goes into
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) with its license text in [`LICENSES/`](LICENSES/). Do not add or
change a license header, an SPDX identifier or a license file without a maintainer's written approval.

## How a pull request is merged

The repository on GitHub is exported from the maintainers' repository. When a maintainer accepts your pull request,
it is merged there and comes back to this repository with you as the author, and the pull request closes.

## Security issues

Do not open a public issue for a vulnerability. Follow [`SECURITY.md`](SECURITY.md).
