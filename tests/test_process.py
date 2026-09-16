import sys

from inferbench.process import run_process


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
