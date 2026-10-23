# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""User commands, with engine imports deferred until calibration starts."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _sizes(value: str) -> list[int]:
    try:
        sizes = sorted({_positive(item) for item in value.split(",")})
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("use comma-separated positive integers, such as 1,8") from exc
    if sizes[-1] > 512:
        raise argparse.ArgumentTypeError("batch sizes must not exceed 512")
    return sizes


def _object(value: str) -> dict:
    try:
        result = json.loads(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a JSON object") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return result


def _engine(requested: str | None) -> str:
    installed = [name for name in ("sglang", "vllm") if importlib.util.find_spec(name) is not None]
    if requested:
        if requested not in installed:
            raise ValueError(f"{requested} is not installed. Install the pinned engine from docs/install.md.")
        return requested
    if len(installed) != 1:
        raise ValueError("choose --engine sglang or --engine vllm in an environment with that engine installed")
    return installed[0]


def _cli_args(options: dict, *, negative_bools: bool = False) -> list[str]:
    result = []
    for name, value in options.items():
        flag = "--" + name.replace("_", "-")
        if value is False:
            if negative_bools:
                result.append("--no-" + name.replace("_", "-"))
            continue
        result.append(flag)
        if value is not True:
            values = value if isinstance(value, list) else [value]
            result.extend(json.dumps(v) if isinstance(v, dict) else str(v) for v in values)
    return result


def serving_command(args: argparse.Namespace) -> list[str]:
    """Return a shell-safe command matching the measured engine settings."""
    if args.engine == "sglang":
        config = {"mode": "serve", "intensity": "chain_intensity", "data_path": str(args.output)}
        options = {
            "model_path": args.model,
            "tp_size": args.tp,
            "host": "127.0.0.1",
            "port": 30000,
            "attention_backend": "flashinfer",
            "page_size": 1,
            "mem_fraction_static": 0.70,
            "context_length": 4608,
            "max_running_requests": max(args.batch_sizes),
            "disable_radix_cache": True,
            "speculative_algorithm": "EAGLE_HARNESS",
            "speculative_draft_model_path": args.draft,
            "speculative_eagle_topk": 1,
            "speculative_num_steps": 7,
            "speculative_num_draft_tokens": 8,
            "speculative_algorithm_config": json.dumps(config),
            # A chain verify on a GDN hybrid target replays the accepted draft inputs into the state at the
            # commit instead of writing one state snapshot per draft token: at batch 64-256 those snapshots
            # are the largest GDN verify cost and their cache takes 36 GB per GPU at width 8.
            "enable_linear_replayssm_spec": True,
        }
        options.update(args.engine_args)
        return [sys.executable, "-m", "sglang.launch_server", *_cli_args(options)]
    config = {
        "method": "eagle_harness",
        "model": args.draft,
        "method_config": {
            "mode": "static",
            "node_top_k": 8,
            "tree_depth": 12,
            "max_verify_width": 32,
            "data_path": str(args.output),
        },
    }
    # max_num_batched_tokens stays vLLM's default, which fits one full batch (see eagle_harness.vllm.calibrate.engine).
    options = {
        "tensor_parallel_size": args.tp,
        "host": "127.0.0.1",
        "port": 8000,
        "attention_backend": "CUSTOM",
        "max_model_len": 4608,
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 1,
        "seed": 0,
    }
    options.update(args.engine_args)
    return [
        "env",
        "VLLM_USE_V2_MODEL_RUNNER=1",
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--speculative-config",
        json.dumps(config),
        *_cli_args(options, negative_bools=True),
    ]


@contextmanager
def _process(command: list[str], log_path: Path):
    # Kill only our process group, including TP workers reparented by the engine.
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield process
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _run(command: list[str], log_path: Path, timeout: int) -> None:
    print(f"Running calibration, log: {log_path}", flush=True)
    with _process(command, log_path) as process:
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"calibration timed out after {timeout}s. Read {log_path}.") from exc
        if code:
            raise RuntimeError(f"calibration exited {code}. Read {log_path}.")


def _vllm(args: argparse.Namespace) -> None:
    common = [
        sys.executable,
        "-m",
        "eagle_harness.vllm.calibrate",
        "--model",
        args.model,
        "--draft",
        args.draft,
        "--tp",
        str(args.tp),
        "--engine-args",
        json.dumps(args.engine_args),
    ]
    raws = []
    # The fitter needs all high-intensity verify buckets, even when serving a narrower tree.
    for width in (8, 16, 32, 64):
        raw = str(args.output / "raw" / f"width-{width}.json")
        _run([*common, "measure", raw, str(width)], args.output / f"width-{width}.log", args.timeout)
        raws.append(raw)
    _run([*common, "fit", str(args.output), *raws], args.output / "fit.log", args.timeout)
    from eagle_harness.vllm.calibrate import validate_tables

    validate_tables(str(args.output))


def _sglang(args: argparse.Namespace) -> None:
    try:
        driver = importlib.import_module("eagle_harness.sglang.calibrate")
    except ModuleNotFoundError as exc:
        if exc.name != "eagle_harness.sglang.calibrate":
            raise
        raise RuntimeError(
            "this wheel lacks the SGLang calibration driver. Install a build with SGLang calibration."
        ) from exc
    data = args.data
    if data is None:
        data = args.output / "smoke-prompts.jsonl"
        prompts = [
            "Explain how a hash table works and give a Python example.",
            "Describe the water cycle in detail for a high school student.",
            "Write a short story about a scientist solving a problem on a remote island.",
            "Compare merge sort and quicksort, including examples and their time complexity.",
        ]
        data.write_text("".join(json.dumps({"category": "smoke", "turns": [p]}) + "\n" for p in prompts))
        print("Using built-in smoke prompts. Supply --data for workload-specific acceptance.", flush=True)
    command = serving_command(args)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command[command.index("--port") + 1] = str(port)
    config_index = command.index("--speculative-algorithm-config") + 1
    config = json.loads(command[config_index])
    config["mode"] = "calibrate"
    command[config_index] = json.dumps(config)
    log = args.output / "server.log"
    with _process(command, log) as process:
        deadline = time.monotonic() + args.timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"SGLang exited before readiness. Read {log}.")
            if time.monotonic() > deadline:
                raise RuntimeError(f"SGLang startup timed out. Read {log}.")
            try:
                with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            time.sleep(1)
        driver.measure(
            port, str(data), args.engine_args.get("tokenizer_path", args.model), args.batch_sizes, args.max_new_tokens
        )
        driver.report(str(args.output), 8, max(args.batch_sizes))
    for file in ("rounds/chain.json", "accept/chain.json"):
        if not (args.output / file).is_file():
            raise RuntimeError(f"SGLang produced no {file}. Read {log}.")


def main(argv: list[str] | None = None) -> int:
    """Calibrate the installed engine, returning nonzero on incomplete measurements."""
    parser = argparse.ArgumentParser(
        prog="eagle-harness", description="Calibrate EAGLE-Harness for your serving setup."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser(
        "calibrate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="measure and fit a calibration, then print the matching serve command",
        description="SGLang measures served chains. vLLM measures batch-one tree costs. "
        "GPU calibration can take longer than five minutes and reserves the selected GPUs.",
    )
    cal.add_argument("--model", required=True, help="target model directory or Hugging Face repository")
    cal.add_argument("--draft", required=True, help="exported draft directory or Hugging Face repository")
    cal.add_argument("--tp", type=_positive, default=1, help="tensor-parallel GPU count")
    cal.add_argument("--engine", choices=("sglang", "vllm"), help="auto-detected if exactly one engine is installed")
    cal.add_argument("--output", type=Path, help="fresh directory (default: ./eagle-calibration-ENGINE)")
    cal.add_argument("--data", type=Path, help="SGLang workload JSONL with category and turns fields")
    cal.add_argument("--batch-sizes", type=_sizes, help="SGLang defaults to 1,8. vLLM supports only 1")
    cal.add_argument("--max-new-tokens", type=_positive, default=1024, help="tokens per SGLang calibration request")
    cal.add_argument(
        "--engine-args", type=_object, default={}, help="engine keyword arguments as JSON, also used for serving"
    )
    cal.add_argument("--timeout", type=_positive, default=1800, help="seconds per vLLM subprocess or SGLang startup")
    cal.add_argument(
        "--dry-run",
        action="store_true",
        help="print settings and serving command without loading models or writing files",
    )
    args = parser.parse_args(argv)

    # Unwind the subprocess context on scheduler termination, not only Ctrl-C.
    def terminate(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        args.engine = args.engine if args.dry_run and args.engine else _engine(args.engine)
        args.batch_sizes = args.batch_sizes or ([1, 8] if args.engine == "sglang" else [1])
        if args.engine == "vllm" and (args.batch_sizes != [1] or args.data):
            raise ValueError("vLLM tree calibration supports only --batch-sizes 1 and does not consume --data")
        reserved = {
            "model",
            "model_path",
            "tp_size",
            "tensor_parallel_size",
            "host",
            "port",
            "worker_extension_cls",
            "max_num_seqs",
            "max_running_requests",
        }
        if any(not key.isidentifier() for key in args.engine_args):
            raise ValueError("--engine-args keys must be Python keyword names with underscores, not CLI flags")
        if any(value is None for value in args.engine_args.values()):
            raise ValueError("--engine-args values must not be null. Omit the option to use its default.")
        conflicts = [key for key in args.engine_args if key in reserved or key.startswith("speculative")]
        if conflicts:
            raise ValueError(f"--engine-args cannot override calibration controls: {conflicts}")
        args.output = (args.output or Path(f"eagle-calibration-{args.engine}")).resolve()
        settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        if args.dry_run:
            print(json.dumps(settings, indent=2))
            print(shlex.join(serving_command(args)))
            return 0
        if args.data and not args.data.is_file():
            raise ValueError(f"workload file does not exist: {args.data}")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "raw").mkdir()
        (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        (_sglang if args.engine == "sglang" else _vllm)(args)
        command = shlex.join(serving_command(args))
        (args.output / "serve.sh").write_text("#!/bin/sh\nset -eu\nexec " + command + "\n")
        print(
            f"Calibration saved in {args.output}. Start the matching server with:\nsh {shlex.quote(str(args.output / 'serve.sh'))}"
        )
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"eagle-harness: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("eagle-harness: interrupted. Partial measurements remain in the output directory.", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
