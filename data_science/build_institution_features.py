import hashlib
import json
from bisect import bisect_left
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean


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

INSTITUTION_HISTORY_RUN = "20261011T145114496743Z"

NEW_FEATURE_NAMES = [
    "institution_prior_quantum_article_count_mean",
    "institution_prior_quantum_article_count_max",
    "institution_recent_quantum_article_count_mean",
    "institution_recent_quantum_article_count_max",
    "institutions_with_no_prior_quantum_articles_fraction",
    "institution_history_features_missing",
]


def three_years_before(value):
    try:
        return value.replace(year=value.year - 3)
    except ValueError:
        return value.replace(year=value.year - 3, day=28)


dataset_bytes = (dataset_folder / "papers_labeled.json").read_bytes()
dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
papers = json.loads(dataset_bytes)
papers_by_id = {paper["openalex_id"]: paper for paper in papers}

if len(papers_by_id) != len(papers):
    raise ValueError("Duplicate paper IDs.")

basic_bytes = (
    dataset_folder / "basic_features" / "features.json"
).read_bytes()
basic = json.loads(basic_bytes)

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

queue_bytes = (
    dataset_folder / "institution_history_queue.json"
).read_bytes()
queue = json.loads(queue_bytes)
queue_hash = hashlib.sha256(queue_bytes).hexdigest()

for source in (basic, splits, queue):
    if source["dataset_sha256"] != dataset_hash:
        raise ValueError("Inputs belong to different datasets.")

jobs_by_id = {
    job["institution_id"]: job for job in queue["jobs"]
}

if len(jobs_by_id) != len(queue["jobs"]):
    raise ValueError("Duplicate institutions in the queue.")

split_by_id = {}

for split_name in ("train", "validation", "test"):
    for paper_id in splits["ids_by_split"][split_name]:
        if paper_id in split_by_id:
            raise ValueError("Duplicate ID in splits.")
        split_by_id[paper_id] = split_name

basic_ids = [row["openalex_id"] for row in basic["rows"]]

if (
    len(basic_ids) != len(set(basic_ids))
    or set(basic_ids) != set(papers_by_id)
    or set(split_by_id) != set(papers_by_id)
):
    raise ValueError("Dataset, feature rows, and splits do not match.")

if basic["target_name"] != "five_year_citation_count":
    raise ValueError("Unexpected target.")


print("Validating institution histories...")

history_folder = (
    dataset_folder
    / "institution_histories"
    / INSTITUTION_HISTORY_RUN
)

histories = {}
history_hasher = hashlib.sha256()
total_excluded = 0

for history_file in sorted(history_folder.glob("I*_history.json")):
    history_bytes = history_file.read_bytes()
    history = json.loads(history_bytes)
    institution_id = history["institution_id"]

    if institution_id not in jobs_by_id or institution_id in histories:
        raise ValueError("Unexpected or duplicate institution history.")

    job = jobs_by_id[institution_id]
    cutoff = date.fromisoformat(job["cutoff_exclusive"])
    works = history["works"]
    work_ids = [work["openalex_id"] for work in works]
    excluded = history.get("excluded_lineage_only_count", 0)

    if (
        history["queue_sha256"] != queue_hash
        or history["validation_status"] != "passed"
        or history["cutoff_exclusive"] != job["cutoff_exclusive"]
        or len(work_ids) != len(set(work_ids))
        or type(excluded) is not int
        or excluded < 0
        or len(works) + excluded != history["api_reported_count"]
    ):
        raise ValueError(f"Invalid history: {history_file.name}")

    dates = []

    for work in works:
        publication_date = date.fromisoformat(work["publication_date"])

        if publication_date >= cutoff:
            raise ValueError("Institution article is outside its window.")

        # Older checkpoints were validated before this field was added.
        if work.get("direct_affiliation_verified", True) is not True:
            raise ValueError("A saved article lacks a verified direct match.")

        if work["primary_topic_id"] not in queue["topic_ids"]:
            raise ValueError("Unexpected topic in institution history.")

        dates.append(publication_date.toordinal())

    histories[institution_id] = {
        "dates": sorted(dates),
        "cutoff": cutoff,
    }

    total_excluded += excluded
    history_hasher.update(history_file.name.encode("utf-8") + b"\0")
    history_hasher.update(history_bytes + b"\0")

if set(histories) != set(jobs_by_id):
    raise ValueError("Institution collection is incomplete.")


print("Building institution features...")

rows = []

for basic_row in basic["rows"]:
    paper_id = basic_row["openalex_id"]
    paper = papers_by_id[paper_id]
    target_date = date.fromisoformat(paper["publication_date"])
    target_ordinal = target_date.toordinal()
    recent_start = three_years_before(target_date).toordinal()

    if (
        basic_row["split"] != split_by_id[paper_id]
        or basic_row["target"] != paper["five_year_citation_count"]
        or set(basic_row["features"]) != set(basic["feature_names"])
    ):
        raise ValueError("Invalid basic feature row.")

    # Count each institution once, even if several authors list it.
    institution_ids = {
        institution["id"]
        for author in paper.get("authors") or []
        for institution in author.get("institutions") or []
        if institution.get("id")
    }

    prior_counts = []
    recent_counts = []

    for institution_id in sorted(institution_ids):
        if institution_id not in histories:
            raise ValueError("Missing institution history.")

        history = histories[institution_id]

        if target_date > history["cutoff"]:
            raise ValueError("History does not cover the target date.")

        dates = history["dates"]

        prior_count = bisect_left(dates, target_ordinal)
        recent_count = prior_count - bisect_left(dates, recent_start)

        prior_counts.append(prior_count)
        recent_counts.append(recent_count)

    new_features = {
        "institution_prior_quantum_article_count_mean": (
            mean(prior_counts) if prior_counts else None
        ),
        "institution_prior_quantum_article_count_max": (
            max(prior_counts) if prior_counts else None
        ),
        "institution_recent_quantum_article_count_mean": (
            mean(recent_counts) if recent_counts else None
        ),
        "institution_recent_quantum_article_count_max": (
            max(recent_counts) if recent_counts else None
        ),
        "institutions_with_no_prior_quantum_articles_fraction": (
            sum(count == 0 for count in prior_counts) / len(prior_counts)
            if prior_counts else None
        ),
        "institution_history_features_missing": int(not institution_ids),
    }

    rows.append({
        "openalex_id": paper_id,
        "split": basic_row["split"],
        "features": {**basic_row["features"], **new_features},
        "target": basic_row["target"],
    })


run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

output_folder = dataset_folder / "basic_institution_features" / run_id
output_folder.mkdir(parents=True, exist_ok=False)

feature_names = basic["feature_names"] + NEW_FEATURE_NAMES

table = {
    "dataset_sha256": dataset_hash,
    "basic_feature_file_sha256": hashlib.sha256(basic_bytes).hexdigest(),
    "institution_history_run": INSTITUTION_HISTORY_RUN,
    "institution_history_queue_sha256": queue_hash,
    "institution_history_files_sha256": history_hasher.hexdigest(),
    "institution_scope": "Direct recorded affiliation",
    "feature_names": feature_names,
    "target_name": basic["target_name"],
    "publication_time_features_verified": False,
    "history_date_rule": "Prior article date < target paper date",
    "recent_activity_window": (
        "target date minus three calendar years <= prior date < target date"
    ),
    "rows": rows,
}

train_rows = [row for row in rows if row["split"] == "train"]

report = {
    "institution_feature_run": run_id,
    "records": len(rows),
    "validated_institution_histories": len(histories),
    "excluded_lineage_only_records": total_excluded,
    "records_by_split": {
        split_name: sum(row["split"] == split_name for row in rows)
        for split_name in ("train", "validation", "test")
    },
    "basic_feature_count": len(basic["feature_names"]),
    "institution_feature_count": len(NEW_FEATURE_NAMES),
    "total_feature_count": len(feature_names),
    "training_null_counts": {
        name: sum(row["features"][name] is None for row in train_rows)
        for name in NEW_FEATURE_NAMES
    },
}

for filename, value in [
    ("features.json", table),
    ("report.json", report),
]:
    (output_folder / filename).write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

print(json.dumps(report, indent=2))
print(f"\nFeatures saved: {output_folder / 'features.json'}")