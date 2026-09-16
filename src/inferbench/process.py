import subprocess
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


ProcessRunner = Callable[[list[str]], ProcessResult]


def run_process(command: list[str]) -> ProcessResult:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    return ProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
