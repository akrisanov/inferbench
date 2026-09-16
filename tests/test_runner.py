import shutil
from pathlib import Path

import pytest

from inferbench.drivers.vllm import VllmBenchDriver
from inferbench.models import Experiment, Target, Workload
from inferbench.process import ProcessResult
from inferbench.runner import (
    BenchmarkExecutionError,
    BenchmarkResultError,
    ExperimentRunner,
)


def create_experiment() -> Experiment:
    return Experiment(
        name="baseline",
        target=Target(
            base_url="http://localhost:8000",
            model="test-model",
        ),
        workload=Workload(
            input_tokens=2048,
            output_tokens=256,
            request_rate=1,
            requests=100,
        ),
    )


def test_run_experiment(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "vllm-result.json"

    def process_runner(command: list[str]) -> ProcessResult:
        result_dir = Path(command[command.index("--result-dir") + 1])
        result_filename = command[command.index("--result-filename") + 1]
        shutil.copyfile(fixture, result_dir / result_filename)

        return ProcessResult(
            returncode=0,
            stdout="benchmark completed",
            stderr="",
        )

    driver = VllmBenchDriver(process_runner=process_runner)
    runner = ExperimentRunner(driver)
    output_dir = tmp_path / "baseline"

    artifact = runner.run(create_experiment(), output_dir)

    assert artifact.experiment.name == "baseline"
    assert artifact.client.completed == 100
    assert artifact.metadata.finished_at >= artifact.metadata.started_at

    assert (output_dir / "vllm-result.json").is_file()
    assert (output_dir / "experiment.json").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "client.json").is_file()


def test_run_fails_when_benchmark_process_fails(tmp_path: Path) -> None:
    def process_runner(command: list[str]) -> ProcessResult:
        return ProcessResult(
            returncode=1,
            stdout="",
            stderr="connection refused",
        )

    driver = VllmBenchDriver(process_runner=process_runner)
    runner = ExperimentRunner(driver)

    with pytest.raises(BenchmarkExecutionError, match="connection refused"):
        runner.run(create_experiment(), tmp_path / "failed")


def test_run_fails_when_result_is_missing(tmp_path: Path) -> None:
    def process_runner(command: list[str]) -> ProcessResult:
        return ProcessResult(
            returncode=0,
            stdout="benchmark completed",
            stderr="",
        )

    driver = VllmBenchDriver(process_runner=process_runner)
    runner = ExperimentRunner(driver)

    with pytest.raises(BenchmarkResultError, match="without producing result file"):
        runner.run(create_experiment(), tmp_path / "missing")
