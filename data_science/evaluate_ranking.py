import hashlib
import json
import math
import statistics
from pathlib import Path


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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_folder",
    choices=[
        "random_forest_basic",
        "catboost_basic",
        "xgboost_basic",
        "lightgbm_basic",
        "catboost_text",
        "catboost_author",
        "catboost_text_author",
        "xgboost_text",
        "lightgbm_text",
        "random_forest_text",
        "catboost_institution",
    ],
)
parser.add_argument("model_run")
args = parser.parse_args()

model_run = args.model_run
experiment_folder = (
    dataset_folder / args.model_folder / model_run
)

dataset_bytes = (dataset_folder / "papers_labeled.json").read_bytes()
dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()

papers = json.loads(dataset_bytes)
papers_by_id = {
    paper["openalex_id"]: paper for paper in papers
}

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

model_report = json.loads(
    (experiment_folder / "report.json").read_text(
        encoding="utf-8"
    )
)

if (
    dataset_hash != splits["dataset_sha256"]
    or dataset_hash != model_report["dataset_sha256"]
):
    raise ValueError("The dataset does not match this experiment.")

predictions = json.loads(
    (experiment_folder / "validation_predictions.json").read_text(
        encoding="utf-8"
    )
)

validation_ids = set(splits["ids_by_split"]["validation"])
prediction_ids = [
    row["openalex_id"] for row in predictions
]

if len(prediction_ids) != len(set(prediction_ids)):
    raise ValueError("Duplicate predictions.")

if set(prediction_ids) != validation_ids:
    raise ValueError("Predictions do not cover exactly validation.")


for row in predictions:
    paper = papers_by_id[row["openalex_id"]]

    if (
        row["actual_five_year_citations"]
        != paper["five_year_citation_count"]
    ):
        raise ValueError("Saved actual citations differ from dataset.")

    prediction = row["predicted_five_year_citations"]

    if not math.isfinite(prediction) or prediction < 0:
        raise ValueError("Invalid prediction.")


K = 10

if len(predictions) < K:
    raise ValueError("Not enough validation papers.")

# Break equal scores consistently using the OpenAlex ID.
predicted_ranking = sorted(
    predictions,
    key=lambda row: (
        -row["predicted_five_year_citations"],
        row["openalex_id"],
    ),
)

actual_ranking = sorted(
    predictions,
    key=lambda row: (
        -row["actual_five_year_citations"],
        row["openalex_id"],
    ),
)

selected = predicted_ranking[:K]
ideal = actual_ranking[:K]

selected_ids = {row["openalex_id"] for row in selected}
ideal_ids = {row["openalex_id"] for row in ideal}

overlap = len(selected_ids & ideal_ids)

selected_citations = sum(
    row["actual_five_year_citations"] for row in selected
)
ideal_citations = sum(
    row["actual_five_year_citations"] for row in ideal
)

validation_mean = statistics.mean(
    row["actual_five_year_citations"] for row in predictions
)

selected_mean = selected_citations / K

report = {
    "model_run": model_run,
    "evaluation_split": "validation",
    "evaluation_scope": "Entire validation set, not daily batches",
    "validation_papers": len(predictions),
    "selection_size": K,
    "tie_break_rule": "OpenAlex ID ascending",
    "overlap_with_actual_top_10": overlap,
    "precision_at_10": overlap / K,
    "random_expected_overlap_with_actual_top_10": (
        K * K / len(predictions)
    ),
    "selected_mean_actual_citations": round(selected_mean, 4),
    "random_expected_mean_actual_citations": round(
        validation_mean, 4
    ),
    "ideal_top_10_mean_actual_citations": round(
        ideal_citations / K, 4
    ),
    "selected_mean_divided_by_random_expected_mean": (
        round(selected_mean / validation_mean, 4)
        if validation_mean > 0 else None
    ),
    "selected_actual_citation_sum": selected_citations,
    "ideal_top_10_actual_citation_sum": ideal_citations,
    "selected_citation_sum_as_fraction_of_ideal": (
        round(selected_citations / ideal_citations, 4)
        if ideal_citations > 0 else None
    ),
    "actual_citations_at_top_10_boundary": (
        ideal[-1]["actual_five_year_citations"]
    ),
    "papers_tied_at_actual_top_10_boundary": sum(
        row["actual_five_year_citations"]
        == ideal[-1]["actual_five_year_citations"]
        for row in predictions
    ),
}

selected_details = []

for rank, row in enumerate(selected, start=1):
    paper = papers_by_id[row["openalex_id"]]

    selected_details.append({
        "predicted_rank": rank,
        "openalex_id": row["openalex_id"],
        "title": paper["title"],
        "predicted_citations": round(
            row["predicted_five_year_citations"], 2
        ),
        "actual_citations": row["actual_five_year_citations"],
        "in_actual_top_10": row["openalex_id"] in ideal_ids,
    })

output = {
    "dataset_sha256": dataset_hash,
    "report": report,
    "selected_papers": selected_details,
}

output_file = experiment_folder / "ranking_report.json"
output_file.write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))
print("\nThe model's selected papers:")
print(json.dumps(selected_details, indent=2, ensure_ascii=False))
print(f"\nReport saved: {output_file}")