import hashlib
import json
from datetime import date
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

dataset_bytes = (dataset_folder / "papers_labeled.json").read_bytes()
dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
papers = json.loads(dataset_bytes)

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

if dataset_hash != splits["dataset_sha256"]:
    raise ValueError("The dataset changed after the splits were created.")


# Associate every paper with its split.
split_by_id = {}

for split_name, paper_ids in splits["ids_by_split"].items():
    for paper_id in paper_ids:
        if paper_id in split_by_id:
            raise ValueError("A paper appears in more than one split.")

        split_by_id[paper_id] = split_name

paper_ids = [paper["openalex_id"] for paper in papers]

if len(paper_ids) != len(set(paper_ids)):
    raise ValueError("Duplicate paper IDs in the dataset.")

if set(paper_ids) != set(split_by_id):
    raise ValueError("The splits do not cover exactly the dataset.")


jobs = {}
missing_author_id_entries = 0

for paper in papers:
    publication_date = paper["publication_date"]
    date.fromisoformat(publication_date)

    # Count an author only once within each paper.
    seen_in_paper = set()

    for author in paper.get("authors") or []:
        author_id = author.get("id")

        if not author_id:
            missing_author_id_entries += 1
            continue

        if author_id in seen_in_paper:
            continue

        seen_in_paper.add(author_id)

        if author_id not in jobs:
            jobs[author_id] = {
                "author_id": author_id,
                "example_name": author.get("name"),
                "earliest_target_date": publication_date,
                "cutoff_exclusive": publication_date,
                "target_paper_count": 0,
                "splits": set(),
            }

        job = jobs[author_id]

        job["earliest_target_date"] = min(
            job["earliest_target_date"], publication_date
        )
        job["cutoff_exclusive"] = max(
            job["cutoff_exclusive"], publication_date
        )
        job["target_paper_count"] += 1
        job["splits"].add(split_by_id[paper["openalex_id"]])


ordered_jobs = []

for author_id in sorted(jobs):
    job = jobs[author_id]
    job["splits"] = sorted(job["splits"])
    ordered_jobs.append(job)

queue = {
    "dataset_sha256": dataset_hash,
    "corpus": "core",
    "work_type": "article",
    "collection_rule": (
        "Collect each author's articles before their latest target date."
    ),
    "feature_rule": (
        "For each target paper, use only history dated strictly "
        "before that paper's publication date."
    ),
    "jobs": ordered_jobs,
}

output_file = dataset_folder / "author_history_queue.json"
output_file.write_text(
    json.dumps(queue, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

report = {
    "papers": len(papers),
    "unique_authors_to_collect": len(ordered_jobs),
    "authors_present_in_each_split": {
        split_name: sum(
            split_name in job["splits"] for job in ordered_jobs
        )
        for split_name in splits["ids_by_split"]
    },
    "authorship_entries_without_author_id": missing_author_id_entries,
}

print(json.dumps(report, indent=2))
print(f"\nQueue saved: {output_file}")