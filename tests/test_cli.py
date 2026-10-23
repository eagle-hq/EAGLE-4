# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The public command must work without importing an engine for help or planning."""

import json
import os
import shlex
import subprocess
import sys
import time

import pytest

from eagle_harness import cli


@pytest.mark.parametrize("command", [[], ["calibrate"]])
def test_help_without_engine_imports(command):
    result = subprocess.run(
        [sys.executable, "-m", "eagle_harness.cli", *command, "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "calibrat" in result.stdout


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_dry_run_is_shell_safe_and_writes_nothing(engine, tmp_path, capsys):
    output = tmp_path / "has spaces"
    model = "model's directory; not a command"
    assert (
        cli.main(
            [
                "calibrate",
                "--engine",
                engine,
                "--model",
                model,
                "--draft",
                "draft with spaces",
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    command = shlex.split(capsys.readouterr().out.splitlines()[-1])
    assert model in command
    assert not output.exists()
    if engine == "vllm":
        spec = json.loads(command[command.index("--speculative-config") + 1])
        assert spec["model"] == "draft with spaces"
        assert spec["method_config"]["data_path"] == str(output)
    else:
        config = json.loads(command[command.index("--speculative-algorithm-config") + 1])
        assert config["mode"] == "serve"
        assert config["intensity"] == "chain_intensity"
        # The chain's GDN verify commits by replay (no per-draft state snapshots); an engine-args False drops it.
        assert "--enable-linear-replayssm-spec" in command


def test_sglang_engine_args_can_drop_the_replay_verify(capsys):
    assert (
        cli.main(
            ["calibrate", "--engine", "sglang", "--model", "m", "--draft", "d", "--dry-run",
             "--engine-args", '{"enable_linear_replayssm_spec": false}']
        )
        == 0
    )
    command = shlex.split(capsys.readouterr().out.splitlines()[-1])
    assert "--enable-linear-replayssm-spec" not in command


@pytest.mark.parametrize(
    "extra", [["--tp", "0"], ["--batch-sizes", "0,1"], ["--batch-sizes", "513"], ["--engine-args", "[]"]]
)
def test_invalid_arguments_fail_before_loading(extra, tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["calibrate", "--model", "m", "--draft", "d", "--engine", "vllm", "--dry-run", *extra])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--batch-sizes", "1,8"],
        ["--data", "x.jsonl"],
        ["--engine-args", '{"model":"other"}'],
        ["--engine-args", '{"speculative_config":{}}'],
    ],
)
def test_refuses_mismatched_calibration_settings(extra, capsys):
    assert cli.main(["calibrate", "--model", "m", "--draft", "d", "--engine", "vllm", "--dry-run", *extra]) == 1
    assert "eagle-harness:" in capsys.readouterr().err


def test_auto_detection_requires_exactly_one_engine(monkeypatch):
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object() if name == "vllm" else None)
    assert cli._engine(None) == "vllm"
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(ValueError, match="choose --engine"):
        cli._engine(None)
    assert cli._engine("sglang") == "sglang"
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="not installed"):
        cli._engine("vllm")


def test_refuses_existing_output(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_engine", lambda name: "vllm")
    sentinel = tmp_path / "keep"
    sentinel.write_text("untouched")
    assert cli.main(["calibrate", "--model", "m", "--draft", "d", "--output", str(tmp_path)]) == 1
    assert sentinel.read_text() == "untouched"


def test_failed_measurement_does_not_publish_serve_command(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_engine", lambda name: "vllm")
    calls = []

    def failed_run(command, log, timeout):
        calls.append(command)
        raise RuntimeError("worker failed")

    monkeypatch.setattr(cli, "_run", failed_run)
    output = tmp_path / "cal"
    assert cli.main(["calibrate", "--model", "m", "--draft", "d", "--output", str(output)]) == 1
    assert len(calls) == 1
    assert not (output / "serve.sh").exists()
    assert json.loads((output / "settings.json").read_text())["model"] == "m"


def test_vllm_sweeps_every_required_width_before_fitting(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_engine", lambda name: "vllm")
    calls = []

    def collect(command, log, timeout):
        calls.append(command)
        if "fit" in command:
            from eagle_harness.core.calibration import AttentionShape
            from eagle_harness.vllm.calibrate import fit, sweep

            def rows(widths):
                return [
                    {"query_length": q, "cache_length": length, "time_us": 100 + 5 * q + length / 100}
                    for q in widths
                    for length in sweep(context_stop=4096)
                ]

            fit(
                str(tmp_path / "cal"),
                rows([1, 8, 16, 32, 64]),
                rows([1, 2, 4, 8]),
                AttentionShape(4),
                AttentionShape(1),
                2,
                context_stop=4096,
            )

    monkeypatch.setattr(cli, "_run", collect)
    output = tmp_path / "cal"
    assert cli.main(["calibrate", "--model", "m", "--draft", "d", "--tp", "2", "--output", str(output)]) == 0
    assert [int(command[-1]) for command in calls[:-1]] == [8, 16, 32, 64]
    assert calls[-1].count("fit") == 1
    assert all(command[command.index("--tp") + 1] == "2" for command in calls)
    assert "exec env VLLM_USE_V2_MODEL_RUNNER=1" in (output / "serve.sh").read_text()


@pytest.mark.skipif(os.name != "posix", reason="GPU workers use POSIX process groups")
def test_sigterm_stops_owned_worker(tmp_path):
    pidfile = tmp_path / "worker.pid"
    worker = f"import os,time; from pathlib import Path; Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(120)"
    program = (
        "from eagle_harness import cli; import sys; "
        "cli._engine=lambda name: 'vllm'; "
        f"cli._vllm=lambda args: cli._run([sys.executable, '-c', {worker!r}], args.output/'test.log', 120); "
        f"raise SystemExit(cli.main(['calibrate','--model','m','--draft','d','--output',{str(tmp_path / 'cal')!r}]))"
    )
    parent = subprocess.Popen([sys.executable, "-c", program], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        parent.terminate()
        parent.communicate(timeout=40)
        assert parent.returncode != 0
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not (tmp_path / "cal/serve.sh").exists()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.communicate(timeout=5)


def test_process_failure_is_not_success(tmp_path):
    with pytest.raises(RuntimeError, match="exited 7"):
        cli._run([sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failure.log", 5)


@pytest.mark.skipif(os.name != "posix", reason="GPU engine workers use POSIX process groups")
def test_process_cleanup_on_exception(tmp_path):
    with (
        pytest.raises(RuntimeError, match="stop"),
        cli._process([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "process.log") as process,
    ):
        pid = process.pid
        raise RuntimeError("stop")
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize(
    "options,expected",
    [
        ({"enable_prefix_caching": False}, ["--no-enable-prefix-caching"]),
        ({"hf_overrides": {"sliding_window": 2048}}, ["--hf-overrides", '{"sliding_window": 2048}']),
    ],
)
def test_vllm_overrides_survive_command_rendering(options, expected, capsys):
    assert (
        cli.main(
            [
                "calibrate",
                "--model",
                "m",
                "--draft",
                "d",
                "--engine",
                "vllm",
                "--engine-args",
                json.dumps(options),
                "--dry-run",
            ]
        )
        == 0
    )
    command = shlex.split(capsys.readouterr().out.splitlines()[-1])
    index = command.index(expected[0])
    assert command[index : index + len(expected)] == expected


@pytest.mark.parametrize("engine_args", [{}, {"max_model_len": 2048}])
def test_vllm_serve_command_token_budget_fits_one_batch(engine_args, capsys):
    # A budget above max_num_seqs * max_model_len makes vLLM's warmup run one request past the block table (#128).
    argv = ["calibrate", "--model", "m", "--draft", "d", "--engine", "vllm", "--dry-run"]
    assert cli.main([*argv, "--engine-args", json.dumps(engine_args)]) == 0
    command = shlex.split(capsys.readouterr().out.splitlines()[-1])

    def flag(name):
        return int(command[command.index(name) + 1]) if name in command else None

    budget = flag("--max-num-batched-tokens")
    assert budget is None or budget <= flag("--max-num-seqs") * flag("--max-model-len")


def test_empty_cost_tables_do_not_publish_serve_command(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_engine", lambda name: "vllm")

    def empty_fit(command, log, timeout):
        if "fit" in command:
            path = tmp_path / "cal/cost/bs_1/model_llm"
            path.mkdir(parents=True)
            (path / "model_interval_0.json").write_text("{}")

    monkeypatch.setattr(cli, "_run", empty_fit)
    assert cli.main(["calibrate", "--model", "m", "--draft", "d", "--output", str(tmp_path / "cal")]) == 1
    assert not (tmp_path / "cal/serve.sh").exists()


def test_engine_option_spelling_cannot_override_reserved_controls(capsys):
    assert (
        cli.main(
            [
                "calibrate",
                "--model",
                "m",
                "--draft",
                "d",
                "--engine",
                "sglang",
                "--dry-run",
                "--engine-args",
                '{"model-path":"other"}',
            ]
        )
        == 1
    )
