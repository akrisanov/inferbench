import json
from datetime import UTC, datetime
from pathlib import Path

from inferbench.artifacts import ExperimentArtifact, RunMetadata, write_artifact
from inferbench.models import Experiment, Target, Workload
from inferbench.results import Percentiles, VllmBenchmarkResult


def create_artifact() -> ExperimentArtifact:
    experiment = Experiment(
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

    metadata = RunMetadata(
        started_at=datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 16, 12, 2, tzinfo=UTC),
    )

    client = VllmBenchmarkResult(
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
        percentiles=Percentiles(
            ttft_ms={99: 91},
            tpot_ms={99: 17},
        ),
    )

    return ExperimentArtifact(
        experiment=experiment,
        metadata=metadata,
        client=client,
    )


def test_experiment_artifact() -> None:
    artifact = create_artifact()

    assert artifact.experiment.name == "baseline"
    assert artifact.metadata.finished_at > artifact.metadata.started_at
    assert artifact.client.completed == 100
    assert artifact.client.percentiles.ttft_ms[99] == 91


def test_write_artifact(tmp_path: Path) -> None:
    artifact = create_artifact()
    output_dir = tmp_path / "baseline"

    write_artifact(artifact, output_dir)

    experiment = json.loads((output_dir / "experiment.json").read_text())
    metadata = json.loads((output_dir / "metadata.json").read_text())
    client = json.loads((output_dir / "client.json").read_text())

    assert experiment["name"] == "baseline"
    assert metadata["started_at"] == "2026-09-16T12:00:00Z"
    assert metadata["finished_at"] == "2026-09-16T12:02:00Z"
    assert client["completed"] == 100
    assert client["percentiles"]["ttft_ms"]["99"] == 91
