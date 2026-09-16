from inferbench.drivers.vllm import VllmBenchDriver
from inferbench.models import Experiment, Target, Workload


def test_build_command() -> None:
    experiment = Experiment(
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

    driver = VllmBenchDriver()

    command = driver.build_command(experiment)

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
    ]
