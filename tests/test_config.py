from pathlib import Path

from inferbench.config import load_experiment


def test_load_experiment() -> None:
    path = Path(__file__).parent / "fixtures" / "basic.yaml"

    experiment = load_experiment(path)

    assert experiment.name == "basic-vllm"
    assert str(experiment.target.base_url) == "http://localhost:8000/"
    assert experiment.target.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert experiment.workload.input_tokens == 2048
    assert experiment.workload.output_tokens == 256
    assert experiment.workload.request_rate == 1
    assert experiment.workload.requests == 100
    assert experiment.workload.seed == 42
