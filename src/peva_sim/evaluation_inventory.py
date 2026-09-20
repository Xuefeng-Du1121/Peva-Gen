"""Load held-out evaluation cases without weakening training provenance."""
import json
from pathlib import Path

from .forcing_scenarios import validate_split_support
from .protocol import file_sha


EVALUATION_SPLITS = ("validation", "test", "iid_test", "ood_test")


def load_evaluation_cases(training_inventory_path, scenario_inventory_path,
                          split, expected_config, expected_field_sha256):
    training_path = Path(training_inventory_path).resolve()
    scenario_path = (training_path if scenario_inventory_path is None else
                     Path(scenario_inventory_path).resolve())
    training = json.loads(training_path.read_text(encoding="utf-8"))
    scenarios = (training if scenario_path == training_path else
                 json.loads(scenario_path.read_text(encoding="utf-8")))
    if scenarios.get("config") != expected_config:
        raise ValueError("Evaluation scenario config differs from training")
    if scenarios.get("field_sha256") != expected_field_sha256:
        raise ValueError("Evaluation scenario field differs from training")
    cases = [row for row in scenarios.get("scenarios", [])
             if row.get("split") == split]
    if not cases:
        raise ValueError("No cases for requested split")
    train_cases = [row for row in training.get("scenarios", [])
                   if row.get("split") == "train"]
    if not train_cases:
        raise ValueError("Training inventory has no training cases")
    validate_split_support(train_cases + cases)
    return cases, {
        "training_inventory_sha256": file_sha(training_path),
        "scenario_inventory_sha256": file_sha(scenario_path),
        "scenario_inventory_path": str(scenario_path),
    }
