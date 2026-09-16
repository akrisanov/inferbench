from inferbench.models import Experiment
from inferbench.process import ProcessResult, ProcessRunner, run_process


class VllmBenchDriver:
    def __init__(self, process_runner: ProcessRunner = run_process) -> None:
        self.process_runner = process_runner

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

    def run(self, experiment: Experiment) -> ProcessResult:
        command = self.build_command(experiment)
        return self.process_runner(command)
