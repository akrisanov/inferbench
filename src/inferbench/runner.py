from datetime import UTC, datetime
from pathlib import Path

from inferbench.artifacts import ExperimentArtifact, RunMetadata, write_artifact
from inferbench.drivers.vllm import VllmBenchDriver
from inferbench.models import Experiment
from inferbench.process import ProcessExecutionError
from inferbench.results import load_vllm_result


class BenchmarkExecutionError(RuntimeError):
    pass


class BenchmarkResultError(RuntimeError):
    pass


class ExperimentRunner:
    def __init__(self, driver: VllmBenchDriver) -> None:
        self.driver = driver

    def run(
        self,
        experiment: Experiment,
        output_dir: Path,
    ) -> ExperimentArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)

        result_path = output_dir / "vllm-result.json"
        started_at = datetime.now(UTC)

        try:
            process_result = self.driver.run(experiment, result_path)
        except ProcessExecutionError as error:
            raise BenchmarkExecutionError(str(error)) from error

        if process_result.returncode != 0:
            raise BenchmarkExecutionError(
                f"Benchmark process failed with exit code {process_result.returncode}: "
                f"{process_result.stderr.strip()}"
            )

        if not result_path.is_file():
            raise BenchmarkResultError(
                f"Benchmark completed without producing result file: {result_path}"
            )

        client_result = load_vllm_result(result_path)
        finished_at = datetime.now(UTC)

        artifact = ExperimentArtifact(
            experiment=experiment,
            metadata=RunMetadata(
                started_at=started_at,
                finished_at=finished_at,
            ),
            client=client_result,
        )

        write_artifact(artifact, output_dir)

        return artifact
