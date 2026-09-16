import pytest
from pydantic import ValidationError

from inferbench.models import Experiment


def test_rejects_unknown_fields() -> None:
    data = {
        "name": "test",
        "target": {
            "base_url": "http://localhost:8000",
            "model": "test-model",
        },
        "workload": {
            "input_tokens": 2048,
            "output_tokens": 256,
            "request_rate": 1,
            "requests": 100,
            "unknown": "value",
        },
    }

    with pytest.raises(ValidationError):
        Experiment.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_tokens", 0),
        ("output_tokens", 0),
        ("request_rate", 0),
        ("requests", 0),
    ],
)
def test_rejects_non_positive_workload_values(field: str, value: int) -> None:
    workload = {
        "input_tokens": 2048,
        "output_tokens": 256,
        "request_rate": 1,
        "requests": 100,
    }
    workload[field] = value

    data = {
        "name": "test",
        "target": {
            "base_url": "http://localhost:8000",
            "model": "test-model",
        },
        "workload": workload,
    }

    with pytest.raises(ValidationError):
        Experiment.model_validate(data)
