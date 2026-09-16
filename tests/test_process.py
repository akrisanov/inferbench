import sys

import pytest

from inferbench.process import ProcessExecutionError, run_process


def test_run_process() -> None:
    result = run_process(
        [
            sys.executable,
            "-c",
            "print('hello inferbench')",
        ]
    )

    assert result.returncode == 0
    assert result.stdout == "hello inferbench\n"
    assert result.stderr == ""


def test_run_process_fails_when_executable_is_missing() -> None:
    with pytest.raises(ProcessExecutionError, match="Failed to execute"):
        run_process(["inferbench-executable-that-does-not-exist"])
