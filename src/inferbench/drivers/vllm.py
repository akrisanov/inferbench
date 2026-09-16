from inferbench.models import Experiment


class VllmBenchDriver:
    def build_command(self, experiment: Experiment) -> list[str]:
        target = experiment.target
        workload = experiment.workload

        return [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "openai",
            "--base-url",
            str(target.base_url).rstrip("/"),
            "--model",
            target.model,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(workload.input_tokens),
            "--random-output-len",
            str(workload.output_tokens),
            "--request-rate",
            str(workload.request_rate),
            "--num-prompts",
            str(workload.requests),
            "--seed",
            str(workload.seed),
            "--ignore-eos",
        ]
