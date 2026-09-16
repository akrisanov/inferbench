import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

PERCENTILE_PATTERN = re.compile(r"^p(\d+)_(ttft|tpot|itl|e2el)_ms$")


class Percentiles(BaseModel):
    ttft_ms: dict[int, float] = Field(default_factory=dict)
    tpot_ms: dict[int, float] = Field(default_factory=dict)
    itl_ms: dict[int, float] = Field(default_factory=dict)
    e2el_ms: dict[int, float] = Field(default_factory=dict)


class VllmBenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    duration: float = Field(gt=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)

    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)

    request_throughput: float = Field(ge=0)
    output_throughput: float = Field(ge=0)
    total_token_throughput: float = Field(ge=0)

    mean_ttft_ms: float = Field(ge=0)
    median_ttft_ms: float = Field(ge=0)

    mean_tpot_ms: float = Field(ge=0)
    median_tpot_ms: float = Field(ge=0)

    mean_itl_ms: float = Field(ge=0)
    median_itl_ms: float = Field(ge=0)

    mean_e2el_ms: float = Field(ge=0)
    median_e2el_ms: float = Field(ge=0)

    percentiles: Percentiles = Field(default_factory=Percentiles)


def _extract_percentiles(data: dict[str, Any]) -> Percentiles:
    percentiles = Percentiles()

    metric_maps = {
        "ttft": percentiles.ttft_ms,
        "tpot": percentiles.tpot_ms,
        "itl": percentiles.itl_ms,
        "e2el": percentiles.e2el_ms,
    }

    for key, value in data.items():
        match = PERCENTILE_PATTERN.match(key)
        if match is None:
            continue

        percentile = int(match.group(1))
        metric = match.group(2)

        metric_maps[metric][percentile] = float(value)

    return percentiles


def load_vllm_result(path: Path) -> VllmBenchmarkResult:
    with path.open(encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)

    return VllmBenchmarkResult.model_validate(
        {
            **data,
            "percentiles": _extract_percentiles(data),
        }
    )
