import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
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

MODELS = {
    "random_forest_text": "20261011T172856075855Z",
    "catboost_text_institution": "20261011T181956463475Z",
}


def citation_group(count):
    # Diagnostic groups only; these are not model inputs.
    if count == 0:
        return "0 citations"
    if count <= 5:
        return "1–5 citations"
    if count <= 20:
        return "6–20 citations"
    if count <= 100:
        return "21–100 citations"
    return "101+ citations"


def metrics(rows):
    actual = [row["actual"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    errors = [
        prediction - target
        for prediction, target in zip(predicted, actual)
    ]
    n = len(rows)

    return {
        "papers": n,
        "mean_actual_citations": round(sum(actual) / n, 4),
        "mean_predicted_citations": round(sum(predicted) / n, 4),
        "mae": round(sum(abs(error) for error in errors) / n, 4),
        "rmse": round(
            math.sqrt(sum(error ** 2 for error in errors) / n), 4
        ),
        "rmsle": round(
            math.sqrt(
                sum(
                    (math.log1p(prediction) - math.log1p(target)) ** 2
                    for prediction, target in zip(predicted, actual)
                ) / n
            ),
            4,
        ),
        # Positive: overprediction on average.
        # Negative: underprediction on average.
        "mean_prediction_minus_actual": round(sum(errors) / n, 4),
    }


dataset_bytes = (dataset_folder / "papers_labeled.json").read_bytes()
dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()

papers = json.loads(dataset_bytes)
papers_by_id = {paper["openalex_id"]: paper for paper in papers}

if len(papers_by_id) != len(papers):
    raise ValueError("Duplicate dataset IDs.")

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

if splits["dataset_sha256"] != dataset_hash:
    raise ValueError("Dataset differs from the split definition.")

validation_list = splits["ids_by_split"]["validation"]
validation_ids = set(validation_list)

if (
    not validation_ids
    or len(validation_ids) != len(validation_list)
    or not validation_ids <= set(papers_by_id)
):
    raise ValueError("Invalid validation IDs.")

results = {}
prediction_hashes = {}

for model_name, model_run in MODELS.items():
    experiment_folder = dataset_folder / model_name / model_run

    report = json.loads(
        (experiment_folder / "report.json").read_text(encoding="utf-8")
    )

    if (
        report["dataset_sha256"] != dataset_hash
        or report["run_id"] != model_run
    ):
        raise ValueError(f"Experiment mismatch: {model_name}")

    prediction_bytes = (
        experiment_folder / "validation_predictions.json"
    ).read_bytes()

    predictions = json.loads(prediction_bytes)
    prediction_ids = [row["openalex_id"] for row in predictions]

    if (
        len(prediction_ids) != len(set(prediction_ids))
        or set(prediction_ids) != validation_ids
    ):
        raise ValueError(f"Invalid prediction coverage: {model_name}")

    prediction_hashes[model_name] = hashlib.sha256(
        prediction_bytes
    ).hexdigest()

    groups = {
        "by_actual_citation_range": defaultdict(list),
        "by_abstract_availability": defaultdict(list),
        "by_observed_topic": defaultdict(list),
    }

    checked_rows = []

    for prediction_row in predictions:
        paper_id = prediction_row["openalex_id"]
        paper = papers_by_id[paper_id]

        actual = paper["five_year_citation_count"]
        predicted = prediction_row["predicted_five_year_citations"]

        if (
            type(actual) is not int
            or actual < 0
            or prediction_row["actual_five_year_citations"] != actual
            or not math.isfinite(predicted)
            or predicted < 0
        ):
            raise ValueError(f"Invalid target or prediction: {paper_id}")

        row = {
            "openalex_id": paper_id,
            "title": paper.get("title"),
            "actual": actual,
            "predicted": predicted,
        }
        checked_rows.append(row)

        abstract_group = (
            "abstract available"
            if (paper.get("abstract") or "").strip()
            else "abstract unavailable"
        )

        topic = (
            (paper.get("observed_primary_topic") or {}).get("name")
            or "unknown"
        )

        groups["by_actual_citation_range"][citation_group(actual)].append(row)
        groups["by_abstract_availability"][abstract_group].append(row)
        groups["by_observed_topic"][topic].append(row)

    largest_errors = sorted(
        checked_rows,
        key=lambda row: (
            -abs(row["predicted"] - row["actual"]),
            row["openalex_id"],
        ),
    )[:10]

    results[model_name] = {
        "model_run": model_run,
        "overall": metrics(checked_rows),
        **{
            dimension: {
                group_name: metrics(group_rows)
                for group_name, group_rows in grouped_rows.items()
            }
            for dimension, grouped_rows in groups.items()
        },
        "ten_largest_absolute_errors": [
            {
                **row,
                "predicted": round(row["predicted"], 4),
                "absolute_error": round(
                    abs(row["predicted"] - row["actual"]), 4
                ),
            }
            for row in largest_errors
        ],
    }


run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output_folder = dataset_folder / "error_analysis" / run_id
output_folder.mkdir(parents=True, exist_ok=False)

output = {
    "dataset_sha256": dataset_hash,
    "prediction_file_sha256": prediction_hashes,
    "evaluation_split": "validation",
    "citation_groups": "0, 1–5, 6–20, 21–100, 101+",
    "bias_definition": "prediction minus actual; negative means underprediction",
    "models": results,
}

output_file = output_folder / "report.json"
output_file.write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

# Print group results; the ten largest errors remain in the saved report.
print(json.dumps(
    {
        model_name: {
            key: value
            for key, value in result.items()
            if key != "ten_largest_absolute_errors"
        }
        for model_name, result in results.items()
    },
    indent=2,
    ensure_ascii=False,
))

print(f"\nFull report: {output_file}")