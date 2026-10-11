import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np


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

MODEL_NAME = "catboost_text_institution_depth4"
MODEL_RUN = "20261011T184918333620Z"

experiment_folder = dataset_folder / MODEL_NAME / MODEL_RUN
model_file = (
    project_folder / "models" / MODEL_NAME / MODEL_RUN / "model.joblib"
)

bundle = joblib.load(model_file)
report = bundle["report"]

feature_file = (
    dataset_folder
    / "basic_text_institution_features"
    / "20261011T181812227144Z"
    / "features.json"
)
feature_bytes = feature_file.read_bytes()
table = json.loads(feature_bytes)

dataset_hash = hashlib.sha256(
    (dataset_folder / "papers_labeled.json").read_bytes()
).hexdigest()

if (
    dataset_hash != report["dataset_sha256"]
    or dataset_hash != table["dataset_sha256"]
    or hashlib.sha256(feature_bytes).hexdigest()
    != report["feature_file_sha256"]
    or bundle["feature_names"] != table["feature_names"]
    or report["run_id"] != MODEL_RUN
    or bundle["target_transform"] != "log1p"
):
    raise ValueError("Model, dataset, or features do not match.")

preprocessor_file = project_folder / report["text_preprocessor_file"]

if (
    hashlib.sha256(preprocessor_file.read_bytes()).hexdigest()
    != report["text_preprocessor_sha256"]
):
    raise ValueError("Text preprocessor hash differs.")

# Verify reproduction without fitting anything.
saved_predictions = json.loads(
    (experiment_folder / "validation_predictions.json").read_text(
        encoding="utf-8"
    )
)
saved_by_id = {
    row["openalex_id"]: row for row in saved_predictions
}

rows = [row for row in table["rows"] if row["split"] == "validation"]
row_ids = [row["openalex_id"] for row in rows]

if (
    len(saved_by_id) != len(saved_predictions)
    or len(row_ids) != len(set(row_ids))
    or set(row_ids) != set(saved_by_id)
):
    raise ValueError("Validation prediction coverage differs.")

inputs = np.array([
    [
        np.nan if row["features"][name] is None
        else row["features"][name]
        for name in bundle["feature_names"]
    ]
    for row in rows
], dtype=float)

reproduced = np.maximum(
    np.expm1(bundle["model"].predict(
        bundle["imputer"].transform(inputs)
    )),
    0,
)

expected = np.array([
    saved_by_id[row["openalex_id"]]["predicted_five_year_citations"]
    for row in rows
])

if not np.allclose(reproduced, expected, rtol=1e-9, atol=1e-9):
    raise ValueError("Saved predictions could not be reproduced.")

# Find a final test report for this exact model.
test_reports = []

for path in (dataset_folder / "final_test").glob("*/report.json"):
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if (
        candidate.get("model_folder") == MODEL_NAME
        and candidate.get("model_run") == MODEL_RUN
        and candidate.get("dataset_sha256") == dataset_hash
    ):
        test_reports.append((path, candidate))

if not test_reports:
    raise ValueError("No matching final test report found.")

test_path, test_report = sorted(test_reports, key=lambda item: str(item[0]))[-1]

model_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()

if model_hash != test_report["model_sha256"]:
    raise ValueError("Model differs from the final tested model.")

# Create an immutable release folder.
package_folder = project_folder / "models" / "releases" / MODEL_RUN
package_folder.mkdir(parents=True, exist_ok=False)

shutil.copy2(model_file, package_folder / "model.joblib")
shutil.copy2(preprocessor_file, package_folder / "text_preprocessor.joblib")
shutil.copy2(test_path, package_folder / "test_report.json")

manifest = {
    "model_folder": MODEL_NAME,
    "model_run": MODEL_RUN,
    "model_file": "model.joblib",
    "model_sha256": model_hash,
    "text_preprocessor_file": "text_preprocessor.joblib",
    "text_preprocessor_sha256": report["text_preprocessor_sha256"],
    "dataset_sha256": dataset_hash,
    "feature_file_sha256": report["feature_file_sha256"],
    "feature_names": bundle["feature_names"],
    "target": "five_year_citation_count",
    "target_transform": "log1p",
    "prediction_conversion": "expm1, then clip to zero",
    "validation_reproduction_passed": True,
    "maximum_prediction_difference": float(
        np.max(np.abs(reproduced - expected))
    ),
    "publication_time_features_verified": False,
    "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
    "versions": report["versions"],
}

(package_folder / "manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

# Record the complete current environment for reproducibility.
environment = subprocess.check_output(
    [sys.executable, "-m", "pip", "freeze"],
    text=True,
)
(package_folder / "training_environment.txt").write_text(
    environment, encoding="utf-8"
)

docs_folder = project_folder / "docs"
docs_folder.mkdir(exist_ok=True)

regression = test_report["regression_metrics"]
ranking = test_report["ranking_metrics"]

model_card = f"""# Quantum Physics Digest: model card

## Purpose
Rank quantum-physics papers by predicted five-year citation impact.
Citation impact is a proxy and does not measure scientific quality.

## Selected model
CatBoostRegressor, depth 4, 500 iterations, learning rate 0.05.
Model run: `{MODEL_RUN}`.
Inputs: 11 basic metadata features, 64 text components,
and 6 institution-history features.

## Data and evaluation
5,000 OpenAlex articles across five quantum topics.
Training: 2015–2017, 3,000 papers.
Validation: 2018, 1,000 papers.
Test: 2019, 1,000 papers.
Preprocessing was fitted on training data only.
Model selection used validation results; test results were assessed afterward.

## Test results
- MAE: {regression["mae"]}
- RMSE: {regression["rmse"]}
- RMSLE: {regression["rmsle"]}
- Precision@10: {ranking["mean_precision_at_10"]}
- NDCG@10 with log citation gain: {ranking["mean_ndcg_at_10_log_citation_gain"]}
- Citation lift over random selection: {ranking["aggregate_citation_lift_over_random"]}

Ranking evaluation uses repeated random pools of 50 papers, selecting 10.
The same 1,000 papers are reused; the 400 pools are not independent observations.
This is not an evaluation on actual daily publication batches.

## Limitations
The model substantially underpredicts highly cited papers.
Test mean actual citations: {regression["mean_actual_citations"]}.
Test mean predicted citations: {regression["mean_predicted_citations"]}.
Metadata and historical links were retrieved retrospectively, rather than
from verified publication-time snapshots.
The chronological split is a retrospective evaluation, not a simulation
of a model deployable in 2019: training labels matured after that date.
Missing metadata, publication-date uncertainty, indexing coverage,
and changes in research trends may affect performance.
Citation forecasts should be presented as uncertain estimates.

## Reproducibility
The release contains the model, fitted imputer, text preprocessor,
feature order, hashes, test report, and an environment snapshot.
Saved validation predictions were reproduced without retraining.
New-paper feature construction and daily collection remain to be implemented.
"""

(docs_folder / "model_card.md").write_text(model_card, encoding="utf-8")

print(json.dumps({
    "validation_reproduction_passed": True,
    "maximum_prediction_difference": manifest["maximum_prediction_difference"],
    "packaged_feature_count": len(bundle["feature_names"]),
    "package_folder": str(package_folder),
    "model_card": str(docs_folder / "model_card.md"),
}, indent=2))