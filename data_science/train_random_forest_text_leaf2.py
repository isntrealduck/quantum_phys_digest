import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def evaluate(actual, predictions):
    errors = actual - predictions

    return {
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "rmse": round(
            float(np.sqrt(np.mean(errors ** 2))), 4
        ),
        "rmsle": round(
            float(
                np.sqrt(
                    np.mean(
                        (
                            np.log1p(actual)
                            - np.log1p(predictions)
                        ) ** 2
                    )
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

feature_bytes = (
    dataset_folder / "basic_text_features" / "20261011T125233037693Z" / "features.json"
).read_bytes()
table = json.loads(feature_bytes)

dataset_hash = hashlib.sha256(
    (dataset_folder / "papers_labeled.json").read_bytes()
).hexdigest()

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

if (
    dataset_hash != table["dataset_sha256"]
    or dataset_hash != splits["dataset_sha256"]
):
    raise ValueError("Dataset, features, and splits do not match.")

feature_names = table["feature_names"]
rows = table["rows"]

row_ids = [row["openalex_id"] for row in rows]

if len(row_ids) != len(set(row_ids)):
    raise ValueError("Duplicate feature rows.")

expected_split_by_id = {}

for split_name, paper_ids in splits["ids_by_split"].items():
    for paper_id in paper_ids:
        if paper_id in expected_split_by_id:
            raise ValueError("A paper appears in multiple splits.")

        expected_split_by_id[paper_id] = split_name

if set(row_ids) != set(expected_split_by_id):
    raise ValueError("Feature rows do not match the splits.")

for row in rows:
    if row["split"] != expected_split_by_id[row["openalex_id"]]:
        raise ValueError("Incorrect split assignment.")

    if set(row["features"]) != set(feature_names):
        raise ValueError("Inconsistent feature columns.")


def get_split(split_name):
    selected = [
        row for row in rows if row["split"] == split_name
    ]

    if not selected:
        raise ValueError(f"Empty split: {split_name}")

    inputs = np.array(
        [
            [
                np.nan if row["features"][name] is None
                else row["features"][name]
                for name in feature_names
            ]
            for row in selected
        ],
        dtype=float,
    )

    targets = np.array(
        [row["target"] for row in selected],
        dtype=float,
    )

    if not np.all(np.isfinite(targets)) or np.any(targets < 0):
        raise ValueError("Invalid targets.")

    return selected, inputs, targets


train_rows, train_inputs, train_targets = get_split("train")
validation_rows, validation_inputs, validation_targets = (
    get_split("validation")
)

forest_settings = {
    "n_estimators": 200,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": 2,
}

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        ),
    ),
    (
        "forest",
        RandomForestRegressor(**forest_settings),
    ),
])

model = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,
    inverse_func=np.expm1,
)

print("Training Random Forest...")
model.fit(train_inputs, train_targets)

# Replacement medians are learned from training data only.
train_predictions = np.maximum(
    model.predict(train_inputs), 0
)
validation_predictions = np.maximum(
    model.predict(validation_inputs), 0
)

if not np.all(np.isfinite(validation_predictions)):
    raise ValueError("The model produced invalid predictions.")

run_id = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%S%fZ"
)

output_folder = (
    dataset_folder / "random_forest_text_leaf2" / run_id
)
output_folder.mkdir(parents=True, exist_ok=False)

model_folder = (
    project_folder / "models" / "random_forest_text_leaf2" / run_id
)
model_folder.mkdir(parents=True, exist_ok=False)

report = {
    "run_id": run_id,
    "text_preprocessor_file": table["text_preprocessor_file"],
    "text_preprocessor_sha256": table["text_preprocessor_sha256"],
    "model": "RandomForestRegressor",
    "target_transform": "log1p",
    "feature_names": feature_names,
    "forest_settings": forest_settings,
    "training_records": len(train_rows),
    "validation_records": len(validation_rows),
    "training_metrics": evaluate(
        train_targets, train_predictions
    ),
    "validation_metrics": evaluate(
        validation_targets, validation_predictions
    ),
    "dataset_sha256": dataset_hash,
    "feature_file_sha256": hashlib.sha256(
        feature_bytes
    ).hexdigest(),
    "versions": {
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    },
}

(output_folder / "report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

predictions = [
    {
        "openalex_id": row["openalex_id"],
        "actual_five_year_citations": row["target"],
        "predicted_five_year_citations": float(prediction),
    }
    for row, prediction in zip(
        validation_rows, validation_predictions
    )
]

(output_folder / "validation_predictions.json").write_text(
    json.dumps(predictions, indent=2),
    encoding="utf-8",
)

joblib.dump(
    {
        "model": model,
        "feature_names": feature_names,
        "report": report,
    },
    model_folder / "model.joblib",
)

print(json.dumps(report, indent=2))
print(f"\nResults: {output_folder}")
print(f"Saved model: {model_folder / 'model.joblib'}")