from pathlib import Path

import typer

from inferbench.config import load_experiment

app = typer.Typer()


@app.callback()
def main() -> None:
    """Benchmark and diagnose LLM serving stacks."""


@app.command()
def run(path: Path) -> None:
    """Load and validate an experiment configuration."""
    experiment = load_experiment(path)

    typer.echo(experiment.model_dump_json(indent=2))
