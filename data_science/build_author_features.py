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

AUTHOR_HISTORY_RUN = "20261011T101956741223Z"

AUTHOR_FEATURE_NAMES = [
    "author_prior_article_count_mean",
    "author_prior_article_count_max",
    "author_recent_article_count_mean",
    "author_recent_article_count_max",
    "author_recorded_career_years_mean",
    "author_recorded_career_years_max",
    "authors_with_no_prior_articles_fraction",
    "author_id_missing_fraction",
    "author_history_features_missing",
]


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def three_years_before(value):
    try:
        return value.replace(year=value.year - 3)
    except ValueError:
        # February 29 becomes February 28.
        return value.replace(year=value.year - 3, day=28)


# Load and verify the dataset, split definition, and basic features.
dataset_bytes = (dataset_folder / "papers_labeled.json").read_bytes()
dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
papers = json.loads(dataset_bytes)

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

basic_bytes = (
    dataset_folder / "basic_features" / "features.json"
).read_bytes()
basic = json.loads(basic_bytes)

queue_bytes = (
    dataset_folder / "author_history_queue.json"
).read_bytes()
queue = json.loads(queue_bytes)
queue_hash = hashlib.sha256(queue_bytes).hexdigest()

for source in (splits, basic, queue):
    if source["dataset_sha256"] != dataset_hash:
        raise ValueError("An input belongs to a different dataset.")

papers_by_id = {paper["openalex_id"]: paper for paper in papers}

if len(papers_by_id) != len(papers):
    raise ValueError("Duplicate paper IDs.")

split_by_id = {}

for split_name in ("train", "validation", "test"):
    for paper_id in splits["ids_by_split"][split_name]:
        if paper_id in split_by_id:
            raise ValueError("Duplicate ID in split definition.")
        split_by_id[paper_id] = split_name

basic_ids = [row["openalex_id"] for row in basic["rows"]]

if (
    len(basic_ids) != len(set(basic_ids))
    or set(basic_ids) != set(papers_by_id)
    or set(split_by_id) != set(papers_by_id)
):
    raise ValueError("Dataset, splits, and feature rows do not match.")

if basic["target_name"] != "five_year_citation_count":
    raise ValueError("Unexpected target name.")

jobs_by_id = {job["author_id"]: job for job in queue["jobs"]}

if len(jobs_by_id) != len(queue["jobs"]):
    raise ValueError("Duplicate authors in the history queue.")


# Validate every saved history and cache publication dates.
history_folder = (
    dataset_folder / "author_histories" / AUTHOR_HISTORY_RUN
)

histories = {}
history_hasher = hashlib.sha256()

print("Validating author histories...")

for history_file in sorted(history_folder.glob("A*_history.json")):
    history_bytes = history_file.read_bytes()
    history = json.loads(history_bytes)
    author_id = history["author_id"]

    if author_id not in jobs_by_id or author_id in histories:
        raise ValueError("Unexpected or duplicate author history.")

    job = jobs_by_id[author_id]
    cutoff = date.fromisoformat(job["cutoff_exclusive"])
    works = history["works"]
    work_ids = [work["openalex_id"] for work in works]

    if (
        history["queue_sha256"] != queue_hash
        or history["validation_status"] != "passed"
        or history["cutoff_exclusive"] != job["cutoff_exclusive"]
        or len(work_ids) != len(set(work_ids))
        or len(works) != history["api_reported_count"]
    ):
        raise ValueError(f"Invalid history: {history_file.name}")

    dates = sorted(
        date.fromisoformat(work["publication_date"]).toordinal()
        for work in works
    )

    if dates and dates[-1] >= cutoff.toordinal():
        raise ValueError("A history contains an article outside its window.")

    histories[author_id] = {
        "dates": dates,
        "cutoff": cutoff,
    }

    history_hasher.update(history_file.name.encode("utf-8") + b"\0")
    history_hasher.update(history_bytes + b"\0")

if set(histories) != set(jobs_by_id):
    raise ValueError("The author-history collection is incomplete.")


# Calculate features separately for each target paper.
rows = []

print("Building author features...")

for basic_row in basic["rows"]:
    paper_id = basic_row["openalex_id"]
    paper = papers_by_id[paper_id]
    target_date = date.fromisoformat(paper["publication_date"])
    target_ordinal = target_date.toordinal()
    recent_start = three_years_before(target_date).toordinal()

    if basic_row["split"] != split_by_id[paper_id]:
        raise ValueError("Incorrect split in a basic feature row.")

    if basic_row["target"] != paper["five_year_citation_count"]:
        raise ValueError("Target differs from the labeled dataset.")

    if set(basic_row["features"]) != set(basic["feature_names"]):
        raise ValueError("Basic feature names do not match the schema.")

    authors = paper.get("authors") or []

    # Count each identified author once within a paper.
    known_ids = sorted({
        author["id"]
        for author in authors
        if author.get("id")
    })

    missing_id_count = sum(
        not author.get("id") for author in authors
    )

    prior_counts = []
    recent_counts = []
    career_lengths = []

    for author_id in known_ids:
        if author_id not in histories:
            raise ValueError(f"Missing history for {author_id}")

        history = histories[author_id]

        if target_date > history["cutoff"]:
            raise ValueError("An author history does not cover the target date.")

        dates = history["dates"]

        # bisect_left excludes articles dated on the target publication day.
        prior_count = bisect_left(dates, target_ordinal)
        recent_count = prior_count - bisect_left(dates, recent_start)

        prior_counts.append(prior_count)
        recent_counts.append(recent_count)

        if prior_count > 0:
            career_lengths.append(
                (target_ordinal - dates[0]) / 365.25
            )

    author_features = {
        "author_prior_article_count_mean": (
            mean(prior_counts) if prior_counts else None
        ),
        "author_prior_article_count_max": (
            max(prior_counts) if prior_counts else None
        ),
        "author_recent_article_count_mean": (
            mean(recent_counts) if recent_counts else None
        ),
        "author_recent_article_count_max": (
            max(recent_counts) if recent_counts else None
        ),
        "author_recorded_career_years_mean": (
            mean(career_lengths) if career_lengths else None
        ),
        "author_recorded_career_years_max": (
            max(career_lengths) if career_lengths else None
        ),
        "authors_with_no_prior_articles_fraction": (
            sum(count == 0 for count in prior_counts) / len(prior_counts)
            if prior_counts else None
        ),
        "author_id_missing_fraction": (
            missing_id_count / len(authors) if authors else None
        ),
        "author_history_features_missing": int(not known_ids),
    }

    rows.append({
        "openalex_id": paper_id,
        "split": basic_row["split"],
        "features": {
            **basic_row["features"],
            **author_features,
        },
        "target": basic_row["target"],
    })


run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
feature_names = basic["feature_names"] + AUTHOR_FEATURE_NAMES

output_folder = dataset_folder / "basic_author_features" / run_id
output_folder.mkdir(parents=True, exist_ok=False)

feature_table = {
    "dataset_sha256": dataset_hash,
    "basic_feature_file_sha256": hashlib.sha256(basic_bytes).hexdigest(),
    "author_history_run": AUTHOR_HISTORY_RUN,
    "author_history_queue_sha256": queue_hash,
    "author_history_files_sha256": history_hasher.hexdigest(),
    "feature_names": feature_names,
    "target_name": basic["target_name"],
    "publication_time_features_verified": False,
    "history_date_rule": "Prior article publication date < target paper date",
    "recent_activity_window": (
        "target date minus three calendar years <= prior date < target date"
    ),
    "rows": rows,
}

training_rows = [row for row in rows if row["split"] == "train"]

report = {
    "author_feature_run": run_id,
    "records": len(rows),
    "records_by_split": {
        split_name: sum(row["split"] == split_name for row in rows)
        for split_name in ("train", "validation", "test")
    },
    "validated_author_histories": len(histories),
    "basic_feature_count": len(basic["feature_names"]),
    "author_feature_count": len(AUTHOR_FEATURE_NAMES),
    "total_feature_count": len(feature_names),
    "author_feature_names": AUTHOR_FEATURE_NAMES,
    "training_null_counts": {
        name: sum(row["features"][name] is None for row in training_rows)
        for name in AUTHOR_FEATURE_NAMES
    },
    "training_papers_without_identified_authors": sum(
        row["features"]["author_history_features_missing"]
        for row in training_rows
    ),
}

save_json(output_folder / "features.json", feature_table)
save_json(output_folder / "report.json", report)

print(json.dumps(report, indent=2))
print(f"\nFeatures saved: {output_folder / 'features.json'}")