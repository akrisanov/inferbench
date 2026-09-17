from pathlib import Path
from typing import Protocol

from inferbench.models import Experiment
from inferbench.process import ProcessResult


class BenchmarkDriver(Protocol):
    def run(
        self,
        experiment: Experiment,
        result_path: Path,
    ) -> ProcessResult: ...
