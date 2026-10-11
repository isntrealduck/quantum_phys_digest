import hashlib
import json
import re
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

TOPIC_IDS = ["T10020", "T10622", "T10682", "T11414", "T11804"]


# Load the dataset and its saved split definition.
dataset_bytes = (dataset_folder / "papers_labeled.json").read_bytes()
papers = json.loads(dataset_bytes)
dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

if dataset_hash != splits["dataset_sha256"]:
    raise ValueError("Dataset differs from the split definition.")


# Map each paper ID to exactly one split.
split_by_id = {}

for split_name in ("train", "validation", "test"):
    for paper_id in splits["ids_by_split"][split_name]:
        if paper_id in split_by_id:
            raise ValueError("A paper appears more than once in the splits.")

        split_by_id[paper_id] = split_name

paper_ids = [paper["openalex_id"] for paper in papers]

if len(paper_ids) != len(set(paper_ids)):
    raise ValueError("The dataset contains duplicate paper IDs.")

if set(paper_ids) != set(split_by_id):
    raise ValueError("Dataset IDs do not match the split IDs.")


institutions = {}
institutions_by_split = {
    "train": set(),
    "validation": set(),
    "test": set(),
}

papers_without_institution_id = 0
institution_entries_without_id = 0


for paper in papers:
    paper_id = paper["openalex_id"]
    split_name = split_by_id[paper_id]
    publication_date = date.fromisoformat(
        paper["publication_date"]
    ).isoformat()

    # An institution can appear under several authors on the same paper.
    # Count that paper only once for each institution.
    paper_institutions = {}

    for author in paper.get("authors") or []:
        for institution in author.get("institutions") or []:
            institution_id = institution.get("id")

            if not institution_id:
                institution_entries_without_id += 1
                continue

            if not re.fullmatch(
                r"https://openalex\.org/I\d+", institution_id
            ):
                raise ValueError(
                    f"Invalid institution ID: {institution_id}"
                )

            if institution_id not in paper_institutions:
                paper_institutions[institution_id] = institution.get("name")

    if not paper_institutions:
        papers_without_institution_id += 1

    for institution_id, name in paper_institutions.items():
        institutions_by_split[split_name].add(institution_id)

        if institution_id not in institutions:
            institutions[institution_id] = {
                "institution_id": institution_id,
                "example_name": name,
                "earliest_target_date": publication_date,
                "cutoff_exclusive": publication_date,
                "target_paper_count": 0,
                "splits": set(),
            }

        job = institutions[institution_id]

        if not job["example_name"] and name:
            job["example_name"] = name

        job["earliest_target_date"] = min(
            job["earliest_target_date"], publication_date
        )

        job["cutoff_exclusive"] = max(
            job["cutoff_exclusive"], publication_date
        )

        job["target_paper_count"] += 1
        job["splits"].add(split_name)


# Convert sets to lists so the queue can be saved as JSON.
jobs = []

for institution_id in sorted(institutions):
    job = dict(institutions[institution_id])
    job["splits"] = sorted(job["splits"])
    jobs.append(job)


queue = {
    "dataset_sha256": dataset_hash,
    "corpus": "core",
    "work_type": "article",
    "topic_ids": TOPIC_IDS,
    "topic_filter": "primary_topic.id",
    "institution_scope": "Direct recorded affiliation",
    "collection_rule": (
        "Collect publications strictly before the institution's "
        "latest target-paper publication date."
    ),
    "feature_rule": (
        "For each target paper, use only institution publications "
        "strictly before that individual paper's publication date."
    ),
    "jobs": jobs,
}

output_file = dataset_folder / "institution_history_queue.json"

output_file.write_text(
    json.dumps(queue, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

report = {
    "papers": len(papers),
    "unique_institutions_to_collect": len(jobs),
    "institutions_present_in_each_split": {
        split_name: len(ids)
        for split_name, ids in institutions_by_split.items()
    },
    "papers_without_recorded_institution_id": (
        papers_without_institution_id
    ),
    "institution_entries_without_id": institution_entries_without_id,
}

print(json.dumps(report, indent=2))
print(f"\nQueue saved: {output_file}")