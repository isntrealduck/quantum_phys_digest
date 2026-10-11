import hashlib
import json
import math
import random
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
FEATURE_RUN = "20261011T181812227144Z"

dataset_hash = hashlib.sha256(
    (dataset_folder / "papers_labeled.json").read_bytes()
).hexdigest()

papers = json.loads(
    (dataset_folder / "papers_labeled.json").read_bytes()
)
papers_by_id = {paper["openalex_id"]: paper for paper in papers}

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

feature_bytes = (
    dataset_folder
    / "basic_text_institution_features"
    / FEATURE_RUN
    / "features.json"
).read_bytes()
table = json.loads(feature_bytes)

model_file = (
    project_folder / "models" / MODEL_NAME / MODEL_RUN / "model.joblib"
)
bundle = joblib.load(model_file)
report = bundle["report"]

if any(
    source["dataset_sha256"] != dataset_hash
    for source in (splits, table, report)
):
    raise ValueError("Dataset hashes do not match.")

if hashlib.sha256(feature_bytes).hexdigest() != report["feature_file_sha256"]:
    raise ValueError("Features differ from those used in training.")

if (
    report["run_id"] != MODEL_RUN
    or bundle["target_transform"] != "log1p"
    or bundle["feature_names"] != table["feature_names"]
):
    raise ValueError("Unexpected model or feature schema.")

preprocessor_file = project_folder / report["text_preprocessor_file"]
if (
    hashlib.sha256(preprocessor_file.read_bytes()).hexdigest()
    != report["text_preprocessor_sha256"]
):
    raise ValueError("Text preprocessor differs from the saved experiment.")

test_list = splits["ids_by_split"]["test"]
test_ids = set(test_list)

rows = [row for row in table["rows"] if row["split"] == "test"]
row_ids = [row["openalex_id"] for row in rows]

if (
    len(test_ids) != len(test_list)
    or len(row_ids) != len(set(row_ids))
    or set(row_ids) != test_ids
):
    raise ValueError("Invalid test coverage.")

names = bundle["feature_names"]

for row in rows:
    if (
        set(row["features"]) != set(names)
        or row["target"]
        != papers_by_id[row["openalex_id"]]["five_year_citation_count"]
    ):
        raise ValueError("Invalid feature columns or target.")

inputs = np.array([
    [
        np.nan if row["features"][name] is None
        else row["features"][name]
        for name in names
    ]
    for row in rows
], dtype=float)

actual = np.array([row["target"] for row in rows], dtype=float)

if (
    np.any(np.isinf(inputs))
    or not np.all(np.isfinite(actual))
    or np.any(actual < 0)
):
    raise ValueError("Invalid inputs or targets.")

clean_inputs = bundle["imputer"].transform(inputs)
predicted = np.maximum(
    np.expm1(bundle["model"].predict(clean_inputs)), 0
)

if not np.all(np.isfinite(predicted)):
    raise ValueError("Invalid predictions.")

errors = predicted - actual

regression_metrics = {
    "mae": round(float(np.mean(np.abs(errors))), 4),
    "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
    "rmsle": round(float(np.sqrt(np.mean(
        (np.log1p(predicted) - np.log1p(actual)) ** 2
    ))), 4),
    "mean_actual_citations": round(float(np.mean(actual)), 4),
    "mean_predicted_citations": round(float(np.mean(predicted)), 4),
}

indexed = {
    row["openalex_id"]: (float(target), float(prediction))
    for row, target, prediction in zip(rows, actual, predicted)
}

# Repeat the same candidate-pool evaluation used on validation.
K = 10
BATCH_SIZE = 50

if len(test_ids) % BATCH_SIZE:
    raise ValueError("Test size must divide evenly into batches.")

precisions = []
ndcgs = []
selected_total = 0
ideal_total = 0
random_total = 0
batch_count = 0


def dcg(ids):
    return sum(
        math.log1p(indexed[paper_id][0]) / math.log2(rank + 2)
        for rank, paper_id in enumerate(ids)
    )


for seed in range(42, 62):
    shuffled = sorted(test_ids)
    random.Random(seed).shuffle(shuffled)

    for start in range(0, len(shuffled), BATCH_SIZE):
        ids = shuffled[start:start + BATCH_SIZE]
        selected = sorted(
            ids, key=lambda paper_id: (-indexed[paper_id][1], paper_id)
        )[:K]
        ideal = sorted(
            ids, key=lambda paper_id: (-indexed[paper_id][0], paper_id)
        )[:K]

        precisions.append(len(set(selected) & set(ideal)) / K)
        ideal_dcg = dcg(ideal)
        if ideal_dcg > 0:
            ndcgs.append(dcg(selected) / ideal_dcg)

        selected_total += sum(indexed[i][0] for i in selected)
        ideal_total += sum(indexed[i][0] for i in ideal)
        random_total += K * sum(indexed[i][0] for i in ids) / BATCH_SIZE
        batch_count += 1

ranking_metrics = {
    "batches": batch_count,
    "mean_precision_at_10": round(float(np.mean(precisions)), 4),
    "random_expected_precision_at_10": K / BATCH_SIZE,
    "mean_ndcg_at_10_log_citation_gain": (
        round(float(np.mean(ndcgs)), 4) if ndcgs else None
    ),
    "mean_selected_actual_citations_per_paper": round(
        selected_total / (batch_count * K), 4
    ),
    "aggregate_citation_lift_over_random": (
        round(selected_total / random_total, 4) if random_total else None
    ),
    "aggregate_selected_citations_as_fraction_of_ideal": (
        round(selected_total / ideal_total, 4) if ideal_total else None
    ),
}

output = {
    "model_folder": MODEL_NAME,
    "model_run": MODEL_RUN,
    "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
    "dataset_sha256": dataset_hash,
    "feature_file_sha256": hashlib.sha256(feature_bytes).hexdigest(),
    "evaluation_split": "test",
    "papers": len(rows),
    "regression_metrics": regression_metrics,
    "ranking_metrics": ranking_metrics,
    "ranking_scope": "Repeated random pools of 50; not real daily batches",
    "dependence_note": "The same test papers are reused across repetitions.",
    "selection_basis": "Validation NDCG, supported by validation MAE.",
}

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output_folder = dataset_folder / "final_test" / run_id
output_folder.mkdir(parents=True, exist_ok=False)

(output_folder / "report.json").write_text(
    json.dumps(output, indent=2), encoding="utf-8"
)

(output_folder / "test_predictions.json").write_text(
    json.dumps([
        {
            "openalex_id": row["openalex_id"],
            "actual_five_year_citations": float(target),
            "predicted_five_year_citations": float(prediction),
        }
        for row, target, prediction in zip(rows, actual, predicted)
    ], indent=2),
    encoding="utf-8",
)

print(json.dumps(output, indent=2))
print(f"\nResults: {output_folder}")