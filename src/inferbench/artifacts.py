from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from inferbench.models import Experiment
from inferbench.results import VllmBenchmarkResult


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    started_at: datetime
    finished_at: datetime


class ExperimentArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: Experiment
    metadata: RunMetadata
    client: VllmBenchmarkResult


def write_artifact(artifact: ExperimentArtifact, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "experiment.json": artifact.experiment,
        "metadata.json": artifact.metadata,
        "client.json": artifact.client,
    }

    for filename, model in files.items():
        path = output_dir / filename
        path.write_text(
            model.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
