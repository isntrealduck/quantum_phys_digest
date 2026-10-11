import hashlib
import json
import math
import random
import statistics
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
    "catboost_text_institution_depth4": "20261011T184918333620Z",
    "catboost_search_20261011T192415401837Z_01": "20261011T192425099738Z",
    "catboost_search_20261011T192415401837Z_02": "20261011T192430541124Z",
    "catboost_search_20261011T192415401837Z_03": "20261011T192436036870Z",
    "catboost_search_20261011T192415401837Z_04": "20261011T192450170116Z",
    "catboost_search_20261011T192415401837Z_05": "20261011T192459312668Z",
    "catboost_search_20261011T192415401837Z_06": "20261011T192504682619Z",
    "catboost_search_20261011T192415401837Z_07": "20261011T192508422984Z",
    "catboost_search_20261011T192415401837Z_08": "20261011T192513317827Z"
}

BATCH_SIZE = 50
K = 10
SEEDS = list(range(42, 62))


def average(values):
    available = [value for value in values if value is not None]
    return statistics.mean(available) if available else None


def rounded(value):
    return round(value, 4) if value is not None else None


# Verify the dataset and validation IDs.
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
    raise ValueError("Dataset differs from the saved splits.")

validation_list = splits["ids_by_split"]["validation"]
validation_ids = set(validation_list)

if (
    len(validation_list) != len(validation_ids)
    or not validation_ids <= set(papers_by_id)
):
    raise ValueError("Invalid validation IDs.")

if not 0 < K < BATCH_SIZE:
    raise ValueError("Selection size must be smaller than batch size.")

if len(validation_ids) % BATCH_SIZE:
    raise ValueError("Validation size must divide evenly into batches.")

actual = {}

for paper_id in validation_ids:
    count = papers_by_id[paper_id]["five_year_citation_count"]

    if type(count) is not int or count < 0:
        raise ValueError("Invalid citation target.")

    actual[paper_id] = count


# Load existing predictions; do not load or retrain models.
predictions_by_model = {}
prediction_hashes = {}

for model_name, model_run in MODELS.items():
    folder = dataset_folder / model_name / model_run

    model_report = json.loads(
        (folder / "report.json").read_text(encoding="utf-8")
    )

    if (
        model_report["dataset_sha256"] != dataset_hash
        or model_report["run_id"] != model_run
    ):
        raise ValueError(f"Experiment mismatch: {model_name}")

    prediction_bytes = (
        folder / "validation_predictions.json"
    ).read_bytes()

    saved_predictions = json.loads(prediction_bytes)
    indexed = {}

    for row in saved_predictions:
        paper_id = row["openalex_id"]

        if paper_id in indexed or paper_id not in validation_ids:
            raise ValueError(f"Unexpected prediction: {model_name}")

        if row["actual_five_year_citations"] != actual[paper_id]:
            raise ValueError("Saved targets differ from the dataset.")

        prediction = row["predicted_five_year_citations"]

        if not math.isfinite(prediction) or prediction < 0:
            raise ValueError("Invalid prediction.")

        indexed[paper_id] = prediction

    if set(indexed) != validation_ids:
        raise ValueError("Predictions do not cover validation exactly.")

    predictions_by_model[model_name] = indexed
    prediction_hashes[model_name] = hashlib.sha256(
        prediction_bytes
    ).hexdigest()


# Construct identical batches for all models.
batches = []

for seed in SEEDS:
    shuffled = sorted(validation_ids)
    random.Random(seed).shuffle(shuffled)

    for start in range(0, len(shuffled), BATCH_SIZE):
        batches.append({
            "seed": seed,
            "batch_number": start // BATCH_SIZE + 1,
            "paper_ids": shuffled[start:start + BATCH_SIZE],
        })


def dcg(ranked_ids):
    # Log citation gain reduces domination by extreme citation counts.
    return sum(
        math.log1p(actual[paper_id]) / math.log2(rank + 2)
        for rank, paper_id in enumerate(ranked_ids)
    )


details = []
summaries = {}

for model_name, model_predictions in predictions_by_model.items():
    model_details = []

    for batch in batches:
        ids = batch["paper_ids"]

        selected = sorted(
            ids,
            key=lambda paper_id: (
                -model_predictions[paper_id], paper_id
            ),
        )[:K]

        ideal = sorted(
            ids,
            key=lambda paper_id: (-actual[paper_id], paper_id),
        )[:K]

        selected_sum = sum(actual[paper_id] for paper_id in selected)
        ideal_sum = sum(actual[paper_id] for paper_id in ideal)
        random_expected_sum = K * statistics.mean(
            actual[paper_id] for paper_id in ids
        )

        ideal_dcg = dcg(ideal)
        overlap = len(set(selected) & set(ideal))

        result = {
            "model": model_name,
            "seed": batch["seed"],
            "batch_number": batch["batch_number"],
            "precision_at_10": overlap / K,
            "ndcg_at_10_log_citation_gain": (
                dcg(selected) / ideal_dcg if ideal_dcg > 0 else None
            ),
            "selected_citation_sum": selected_sum,
            "ideal_citation_sum": ideal_sum,
            "random_expected_citation_sum": random_expected_sum,
            "selected_fraction_of_ideal": (
                selected_sum / ideal_sum if ideal_sum > 0 else None
            ),
            "selected_ids": selected,
        }

        model_details.append(result)

    selected_total = sum(
        row["selected_citation_sum"] for row in model_details
    )
    ideal_total = sum(
        row["ideal_citation_sum"] for row in model_details
    )
    random_total = sum(
        row["random_expected_citation_sum"] for row in model_details
    )

    summaries[model_name] = {
        "model_run": MODELS[model_name],
        "evaluated_batches": len(model_details),
        "mean_precision_at_10": rounded(average(
            [row["precision_at_10"] for row in model_details]
        )),
        "mean_ndcg_at_10_log_citation_gain": rounded(average(
            [
                row["ndcg_at_10_log_citation_gain"]
                for row in model_details
            ]
        )),
        "mean_selected_actual_citations_per_paper": rounded(
            selected_total / (len(model_details) * K)
        ),
        "aggregate_citation_lift_over_random": rounded(
            selected_total / random_total if random_total > 0 else None
        ),
        "aggregate_selected_citations_as_fraction_of_ideal": rounded(
            selected_total / ideal_total if ideal_total > 0 else None
        ),
        "batches_without_positive_citations": sum(
            row["ideal_citation_sum"] == 0 for row in model_details
        ),
    }

    details.extend(model_details)


run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output_folder = dataset_folder / "batch_ranking" / run_id
output_folder.mkdir(parents=True, exist_ok=False)

output = {
    "dataset_sha256": dataset_hash,
    "prediction_file_sha256": prediction_hashes,
    "evaluation_split": "validation",
    "scope": "Repeated random candidate pools; not real daily batches",
    "batch_size": BATCH_SIZE,
    "selection_size": K,
    "seeds": SEEDS,
    "tie_break_rule": "OpenAlex ID ascending",
    "ndcg_gain": "log1p(actual five-year citations)",
    "random_expected_precision_at_10": K / BATCH_SIZE,
    "dependence_note": (
        "The same validation papers are reused across repetitions. "
        "The batches are not independent observations."
    ),
    "summaries": summaries,
    "batches": batches,
    "batch_results": details,
}

(output_folder / "report.json").write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps({
    "validation_papers": len(validation_ids),
    "batch_size": BATCH_SIZE,
    "selection_size": K,
    "grouping_repetitions": len(SEEDS),
    "batches_per_model": len(batches),
    "random_expected_precision_at_10": K / BATCH_SIZE,
    "models": summaries,
}, indent=2))

print(f"\nFull report: {output_folder / 'report.json'}")