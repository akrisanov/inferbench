from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from inferbench.artifacts import ExperimentArtifact, RunMetadata
from inferbench.cli import app
from inferbench.models import Experiment
from inferbench.results import Percentiles, VllmBenchmarkResult
from inferbench.runner import ExperimentRunner

runner = CliRunner()


def test_run_command(monkeypatch, tmp_path: Path) -> None:
    def fake_run(
        _self: ExperimentRunner,
        experiment: Experiment,
        _output_dir: Path,
    ) -> ExperimentArtifact:
        return ExperimentArtifact(
            experiment=experiment,
            metadata=RunMetadata(
                started_at=datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
                finished_at=datetime(2026, 9, 16, 12, 2, tzinfo=UTC),
            ),
            client=VllmBenchmarkResult(
                duration=120,
                completed=100,
                failed=0,
                total_input_tokens=204800,
                total_output_tokens=25600,
                request_throughput=0.83,
                output_throughput=213.33,
                total_token_throughput=1920,
                mean_ttft_ms=42,
                median_ttft_ms=38,
                mean_tpot_ms=12,
                median_tpot_ms=11,
                mean_itl_ms=12,
                median_itl_ms=11,
                mean_e2el_ms=3100,
                median_e2el_ms=3000,
                percentiles=Percentiles(),
            ),
        )

    monkeypatch.setattr(ExperimentRunner, "run", fake_run)

    result = runner.invoke(
        app,
        [
            "run",
            "tests/fixtures/basic.yaml",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Experiment completed: basic-vllm" in result.stdout
    assert f"Artifacts: {tmp_path / 'basic-vllm'}" in result.stdout
