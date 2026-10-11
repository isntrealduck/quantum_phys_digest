import hashlib
import json
from datetime import date
from pathlib import Path
from statistics import median


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

history_folder = (
    dataset_folder / "author_histories" / AUTHOR_HISTORY_RUN
)

queue_bytes = (
    dataset_folder / "author_history_queue.json"
).read_bytes()

queue = json.loads(queue_bytes)
queue_hash = hashlib.sha256(queue_bytes).hexdigest()

dataset_hash = hashlib.sha256(
    (dataset_folder / "papers_labeled.json").read_bytes()
).hexdigest()

if dataset_hash != queue["dataset_sha256"]:
    raise ValueError("Dataset differs from the author queue.")

jobs_by_id = {job["author_id"]: job for job in queue["jobs"]}

if len(jobs_by_id) != len(queue["jobs"]):
    raise ValueError("Duplicate authors in the queue.")

history_files = sorted(history_folder.glob("A*_history.json"))

if not history_files:
    raise SystemExit("No completed author history files were found.")

completed_authors = set()
articles = {}
article_counts = []
author_article_links = 0


for history_file in history_files:
    history = json.loads(
        history_file.read_text(encoding="utf-8")
    )

    author_id = history["author_id"]

    if author_id not in jobs_by_id or author_id in completed_authors:
        raise ValueError(f"Unexpected or duplicate author: {author_id}")

    job = jobs_by_id[author_id]
    works = history["works"]
    work_ids = [work["openalex_id"] for work in works]
    cutoff = date.fromisoformat(job["cutoff_exclusive"])

    if (
        history["queue_sha256"] != queue_hash
        or history["cutoff_exclusive"] != job["cutoff_exclusive"]
        or history["validation_status"] != "passed"
        or len(work_ids) != len(set(work_ids))
        or len(works) != history["api_reported_count"]
    ):
        raise ValueError(f"Invalid history: {history_file.name}")

    completed_authors.add(author_id)
    article_counts.append(len(works))
    author_article_links += len(works)

    for work in works:
        work_id = work["openalex_id"]
        publication_date = date.fromisoformat(
            work["publication_date"]
        ).isoformat()

        if date.fromisoformat(publication_date) >= cutoff:
            raise ValueError("An earlier article is outside its window.")

        if work_id not in articles:
            articles[work_id] = {
                "publication_date": publication_date,
                "author_ids": set(),
                "cutoffs": set(),
            }

        article = articles[work_id]

        if article["publication_date"] != publication_date:
            raise ValueError(
                f"Inconsistent publication dates for {work_id}"
            )

        article["author_ids"].add(author_id)
        article["cutoffs"].add(cutoff.isoformat())


report = {
    "author_history_run": AUTHOR_HISTORY_RUN,
    "completed_author_histories": len(completed_authors),
    "authors_in_full_queue": len(jobs_by_id),
    "collection_complete": completed_authors == set(jobs_by_id),
    "author_article_links": author_article_links,
    "distinct_earlier_articles": len(articles),
    "repeated_article_links_avoided_by_deduplication": (
        author_article_links - len(articles)
    ),
    "articles_shared_by_multiple_collected_authors": sum(
        len(article["author_ids"]) > 1
        for article in articles.values()
    ),
    "distinct_article_and_author_cutoff_pairs": sum(
        len(article["cutoffs"])
        for article in articles.values()
    ),
    "earlier_articles_per_completed_author": {
        "minimum": min(article_counts),
        "median": median(article_counts),
        "maximum": max(article_counts),
        "authors_with_zero": sum(count == 0 for count in article_counts),
    },
    "minimum_requests_for_one_query_per_distinct_article": len(articles),
    "request_estimate_note": (
        "A lower bound for the per-article collection approach. "
        "Pagination and retries require additional requests. "
        "Counts cover completed author histories only."
    ),
}

output_file = dataset_folder / "author_citation_workload_report.json"

output_file.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))
print(f"\nReport saved: {output_file}")