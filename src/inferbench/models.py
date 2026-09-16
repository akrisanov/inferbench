from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: HttpUrl
    model: str


class Workload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    request_rate: float = Field(gt=0)
    requests: int = Field(gt=0)
    seed: int = 42


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    target: Target
    workload: Workload
