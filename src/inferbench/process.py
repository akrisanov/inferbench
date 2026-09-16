import subprocess
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


ProcessRunner = Callable[[list[str]], ProcessResult]


class ProcessExecutionError(RuntimeError):
    pass


def run_process(command: list[str]) -> ProcessResult:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ProcessExecutionError(f"Failed to execute {command[0]!r}: {error}") from error

    return ProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
