from typer.testing import CliRunner

from inferbench.cli import app

runner = CliRunner()


def test_run_command() -> None:
    result = runner.invoke(app, ["run", "tests/fixtures/basic.yaml"])

    assert result.exit_code == 0
    assert '"name": "basic-vllm"' in result.stdout
    assert '"model": "meta-llama/Llama-3.1-8B-Instruct"' in result.stdout
