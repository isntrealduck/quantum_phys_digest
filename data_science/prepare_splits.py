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

dataset_file = dataset_folder / "papers_labeled.json"
dataset_bytes = dataset_file.read_bytes()
papers = json.loads(dataset_bytes)

years_by_split = {
    "train": [2015, 2016, 2017],
    "validation": [2018],
    "test": [2019],
}

splits = {}

for split_name, years in years_by_split.items():
    splits[split_name] = sorted(
        paper["openalex_id"]
        for paper in papers
        if paper["publication_year"] in years
    )

all_ids = [
    paper_id
    for split_ids in splits.values()
    for paper_id in split_ids
]

dataset_ids = [paper["openalex_id"] for paper in papers]

if len(dataset_ids) != len(set(dataset_ids)):
    raise ValueError("Duplicate IDs in the dataset.")

if len(all_ids) != len(set(all_ids)):
    raise ValueError("A paper appears in multiple splits.")

if set(all_ids) != set(dataset_ids):
    raise ValueError("Some papers were not assigned to a split.")

expected_counts = {
    "train": 3000,
    "validation": 1000,
    "test": 1000,
}

for split_name, expected in expected_counts.items():
    if len(splits[split_name]) != expected:
        raise ValueError(f"Unexpected size for {split_name}.")

output = {
    "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
    "target": "five_year_citation_count",
    "split_method": "publication_year",
    "evaluation_type": "retrospective chronological holdout",
    "years_by_split": years_by_split,
    "ids_by_split": splits,
}

output_file = dataset_folder / "splits.json"
output_file.write_text(
    json.dumps(output, indent=2),
    encoding="utf-8",
)

for split_name, split_ids in splits.items():
    print(f"{split_name}: {len(split_ids)} papers")

print(f"\nSplits saved: {output_file}")