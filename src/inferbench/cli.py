from pathlib import Path
from typing import Annotated

import typer

from inferbench.config import load_experiment
from inferbench.drivers.vllm import VllmBenchDriver
from inferbench.runner import BenchmarkExecutionError, BenchmarkResultError, ExperimentRunner

app = typer.Typer()


@app.callback()
def main() -> None:
    """Benchmark and diagnose LLM serving stacks."""


@app.command()
def run(
    path: Path,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for experiment artifacts.",
        ),
    ] = Path("results"),
) -> None:
    """Run an experiment."""
    experiment = load_experiment(path)

    run_dir = output_dir / experiment.name
    runner = ExperimentRunner(VllmBenchDriver())

    try:
        artifact = runner.run(experiment, run_dir)
    except (BenchmarkExecutionError, BenchmarkResultError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Experiment completed: {artifact.experiment.name}")
    typer.echo(f"Artifacts: {run_dir}")
