import hashlib
import json
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
papers = json.loads(dataset_bytes)

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

if hashlib.sha256(dataset_bytes).hexdigest() != splits["dataset_sha256"]:
    raise ValueError("Dataset differs from the split definition.")

train_ids = set(splits["ids_by_split"]["train"])

train = sorted(
    (
        paper for paper in papers
        if paper["openalex_id"] in train_ids
    ),
    key=lambda paper: (
        paper["publication_date"],
        paper["openalex_id"],
    ),
)

if len(train) != len(train_ids):
    raise ValueError("Training records are missing.")

author_ids = set()
institution_ids = set()
authorships_without_id = 0
pilot_authors = []

for paper in train:
    for author in paper.get("authors") or []:
        author_id = author.get("id")

        if author_id:
            if author_id not in author_ids and len(pilot_authors) < 5:
                pilot_authors.append({
                    "author_id": author_id,
                    "name": author.get("name"),
                    "example_paper_id": paper["openalex_id"],
                    "example_publication_date": paper["publication_date"],
                })

            author_ids.add(author_id)
        else:
            authorships_without_id += 1

        for institution in author.get("institutions") or []:
            if institution.get("id"):
                institution_ids.add(institution["id"])

report = {
    "split": "train",
    "papers": len(train),
    "unique_recorded_author_ids": len(author_ids),
    "unique_recorded_institution_ids": len(institution_ids),
    "authorship_entries_without_author_id": authorships_without_id,
    "pilot_authors": pilot_authors,
}

output_file = dataset_folder / "historical_entity_inventory.json"
output_file.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(report, indent=2, ensure_ascii=False))