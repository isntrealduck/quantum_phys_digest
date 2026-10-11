import hashlib
import json
import math
import statistics
from pathlib import Path


def evaluate(actual, prediction):
    """Evaluate one constant prediction against all validation papers."""
    errors = [value - prediction for value in actual]

    return {
        "prediction": round(prediction, 4),
        "mae": round(
            statistics.mean(abs(error) for error in errors), 4
        ),
        "rmse": round(
            math.sqrt(
                statistics.mean(error ** 2 for error in errors)
            ),
            4,
        ),
        "rmsle": round(
            math.sqrt(
                statistics.mean(
                    (
                        math.log1p(value)
                        - math.log1p(prediction)
                    ) ** 2
                    for value in actual
                )
            ),
            4,
        ),
    }


project_folder = Path(__file__).resolve().parent.parent

dataset_folder = (
    project_folder
    / "data"
    / "processed"
    / "openalex_historical"
    / "20261010T202611867850Z"
    / "labeled"
    / "20261010T204724059047Z"
)

feature_file = dataset_folder / "basic_features" / "features.json"
feature_table = json.loads(
    feature_file.read_text(encoding="utf-8")
)

dataset_hash = hashlib.sha256(
    (dataset_folder / "papers_labeled.json").read_bytes()
).hexdigest()

if dataset_hash != feature_table["dataset_sha256"]:
    raise ValueError("The feature table refers to a different dataset.")

rows = feature_table["rows"]

train_targets = [
    row["target"] for row in rows if row["split"] == "train"
]
validation_targets = [
    row["target"] for row in rows
    if row["split"] == "validation"
]

if not train_targets or not validation_targets:
    raise ValueError("Training or validation data is empty.")

for value in train_targets + validation_targets:
    if type(value) is not int or value < 0:
        raise ValueError("Invalid citation target.")


# Learn every prediction from training targets only.
predictions = {
    "training_median": statistics.median(train_targets),
    "training_mean": statistics.mean(train_targets),
    "training_log_mean": math.expm1(
        statistics.mean(
            math.log1p(value) for value in train_targets
        )
    ),
}

report = {
    "training_records": len(train_targets),
    "validation_records": len(validation_targets),
    "evaluation_split": "validation",
    "baselines": {
        name: evaluate(validation_targets, prediction)
        for name, prediction in predictions.items()
    },
}

output_folder = dataset_folder / "baseline_results"
output_folder.mkdir(parents=True, exist_ok=True)

(output_folder / "validation_baselines.json").write_text(
    json.dumps(
        {
            "dataset_sha256": dataset_hash,
            **report,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))