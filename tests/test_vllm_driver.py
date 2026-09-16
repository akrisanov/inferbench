from pathlib import Path

from inferbench.drivers.vllm import VllmBenchDriver
from inferbench.models import Experiment, Target, Workload
from inferbench.process import ProcessResult


def create_experiment() -> Experiment:
    return Experiment(
        name="basic-vllm",
        target=Target(
            base_url="http://localhost:8000",
            model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        workload=Workload(
            input_tokens=2048,
            output_tokens=256,
            request_rate=1,
            requests=100,
            seed=42,
        ),
    )


def test_build_command() -> None:
    experiment = create_experiment()
    result_path = Path("results/benchmark.json")
    driver = VllmBenchDriver()

    command = driver.build_command(experiment, result_path)

    assert command == [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        "http://localhost:8000",
        "--model",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--dataset-name",
        "random",
        "--random-input-len",
        "2048",
        "--random-output-len",
        "256",
        "--request-rate",
        "1.0",
        "--num-prompts",
        "100",
        "--seed",
        "42",
        "--ignore-eos",
        "--save-result",
        "--result-dir",
        "results",
        "--result-filename",
        "benchmark.json",
    ]


def test_run_executes_benchmark_command() -> None:
    experiment = create_experiment()
    result_path = Path("results/benchmark.json")
    executed_command: list[str] = []

    def process_runner(command: list[str]) -> ProcessResult:
        executed_command.extend(command)

        return ProcessResult(
            returncode=0,
            stdout="benchmark completed",
            stderr="",
        )

    driver = VllmBenchDriver(process_runner=process_runner)

    result = driver.run(experiment, result_path)

    assert executed_command == driver.build_command(experiment, result_path)
    assert result.returncode == 0
    assert result.stdout == "benchmark completed"
    assert result.stderr == ""
