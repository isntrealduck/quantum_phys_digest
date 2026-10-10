import json
from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parent
COLLECTION_RUN = "20261010T184050630942Z"
CITATION_RUN = "20261010T195747585977Z"

collection_folder = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "openalex_historical"
    / COLLECTION_RUN
)

labels_folder = (
    collection_folder
    / "citation_labels"
    / CITATION_RUN
)

papers = json.loads(
    (collection_folder / "papers.json").read_text(encoding="utf-8")
)

labels = json.loads(
    (labels_folder / "labels.json").read_text(encoding="utf-8")
)

quality_report = json.loads(
    (labels_folder / "label_quality_report.json").read_text(encoding="utf-8")
)

if (
    quality_report.get("all_checks_passed") is not True
    or quality_report.get("collection_run") != COLLECTION_RUN
    or quality_report.get("citation_run") != CITATION_RUN
):
    raise ValueError("The quality report does not approve these runs.")

paper_ids = [paper["openalex_id"] for paper in papers]
label_ids = [label["openalex_id"] for label in labels]

if len(paper_ids) != len(set(paper_ids)):
    raise ValueError("Duplicate paper IDs.")

if len(label_ids) != len(set(label_ids)):
    raise ValueError("Duplicate label IDs.")

if set(paper_ids) != set(label_ids):
    raise ValueError("Paper IDs and label IDs do not match.")

labels_by_id = {
    label["openalex_id"]: label
    for label in labels
}

dataset = []

for paper in papers:
    label = labels_by_id[paper["openalex_id"]]
    count = label["five_year_citation_count"]

    if type(count) is not int or count < 0:
        raise ValueError("Invalid citation target.")

    # Create a copy so the original paper record stays unchanged.
    record = dict(paper)
    record["five_year_citation_count"] = count

    record["target_provenance"] = {
        "citation_run": CITATION_RUN,
        "window_start_inclusive": label["publication_date"],
        "window_end_exclusive": label["window_end_exclusive"],
        "corpus": "core",
        "citing_works_file": (
            labels_folder / label["citing_works_file"]
        ).relative_to(PROJECT_FOLDER).as_posix(),
    }

    dataset.append(record)

# Keep datasets from different citation runs separate.
output_folder = collection_folder / "labeled" / CITATION_RUN
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "papers_labeled.json"
output_file.write_text(
    json.dumps(dataset, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Labeled records: {len(dataset)}")
print(f"Unique OpenAlex IDs: {len(set(paper_ids))}")
print(f"Dataset saved: {output_file}")