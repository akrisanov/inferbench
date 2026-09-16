from pathlib import Path
from typing import Any

import yaml

from inferbench.models import Experiment


def load_experiment(path: Path) -> Experiment:
    with path.open(encoding="utf-8") as file:
        data: Any = yaml.safe_load(file)

    return Experiment.model_validate(data)
