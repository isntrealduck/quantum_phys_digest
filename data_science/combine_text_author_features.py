import hashlib
import json
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

text_file = (
    dataset_folder
    / "basic_text_features"
    / "20261011T125233037693Z"
    / "features.json"
)

author_file = (
    dataset_folder
    / "basic_author_features"
    / "20261011T163256625581Z"
    / "features.json"
)

text_bytes = text_file.read_bytes()
author_bytes = author_file.read_bytes()

text = json.loads(text_bytes)
author = json.loads(author_bytes)

dataset_hash = hashlib.sha256(
    (dataset_folder / "papers_labeled.json").read_bytes()
).hexdigest()

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

if any(
    source["dataset_sha256"] != dataset_hash
    for source in (text, author, splits)
):
    raise ValueError("Inputs belong to different datasets.")

if text["target_name"] != author["target_name"]:
    raise ValueError("Target names differ.")

if (
    text["basic_feature_file_sha256"]
    != author["basic_feature_file_sha256"]
):
    raise ValueError("Inputs use different basic features.")


def index_rows(table):
    indexed = {}

    if len(table["feature_names"]) != len(set(table["feature_names"])):
        raise ValueError("Duplicate feature names.")

    for row in table["rows"]:
        paper_id = row["openalex_id"]

        if paper_id in indexed:
            raise ValueError("Duplicate paper ID.")

        if set(row["features"]) != set(table["feature_names"]):
            raise ValueError("Row features do not match the schema.")

        indexed[paper_id] = row

    return indexed


text_rows = index_rows(text)
author_rows = index_rows(author)

split_by_id = {}

for split_name in ("train", "validation", "test"):
    for paper_id in splits["ids_by_split"][split_name]:
        if paper_id in split_by_id:
            raise ValueError("Duplicate paper ID in splits.")
        split_by_id[paper_id] = split_name

if not (
    set(text_rows) == set(author_rows) == set(split_by_id)
):
    raise ValueError("Paper IDs do not match.")

# Verify the saved text preprocessor before referencing it.
preprocessor_file = project_folder / text["text_preprocessor_file"]

if (
    hashlib.sha256(preprocessor_file.read_bytes()).hexdigest()
    != text["text_preprocessor_sha256"]
):
    raise ValueError("Text preprocessor hash does not match.")


# Shared basic features appear only once.
extra_author_names = [
    name for name in author["feature_names"]
    if name not in text["feature_names"]
]

feature_names = text["feature_names"] + extra_author_names
shared_names = set(text["feature_names"]) & set(author["feature_names"])

rows = []

for text_row in text["rows"]:
    paper_id = text_row["openalex_id"]
    author_row = author_rows[paper_id]

    if (
        text_row["split"] != split_by_id[paper_id]
        or author_row["split"] != split_by_id[paper_id]
        or text_row["target"] != author_row["target"]
    ):
        raise ValueError("Split or target mismatch.")

    for name in shared_names:
        if text_row["features"][name] != author_row["features"][name]:
            raise ValueError(f"Shared feature differs: {name}")

    rows.append({
        "openalex_id": paper_id,
        "split": text_row["split"],
        "features": {
            **text_row["features"],
            **{
                name: author_row["features"][name]
                for name in extra_author_names
            },
        },
        "target": text_row["target"],
    })


run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

output_folder = dataset_folder / "basic_text_author_features" / run_id
output_folder.mkdir(parents=True, exist_ok=False)

combined = {
    "dataset_sha256": dataset_hash,
    "feature_names": feature_names,
    "target_name": text["target_name"],
    "publication_time_features_verified": False,
    "text_preprocessor_file": text["text_preprocessor_file"],
    "text_preprocessor_sha256": text["text_preprocessor_sha256"],
    "author_history_run": author["author_history_run"],
    "author_history_files_sha256": author["author_history_files_sha256"],
    "text_feature_file_sha256": hashlib.sha256(text_bytes).hexdigest(),
    "author_feature_file_sha256": hashlib.sha256(author_bytes).hexdigest(),
    "rows": rows,
}

(output_folder / "features.json").write_text(
    json.dumps(combined, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps({
    "combined_feature_run": run_id,
    "records": len(rows),
    "total_feature_count": len(feature_names),
    "additional_author_features": len(extra_author_names),
    "records_by_split": {
        split_name: sum(row["split"] == split_name for row in rows)
        for split_name in ("train", "validation", "test")
    },
}, indent=2))

print(f"\nFeatures saved: {output_folder / 'features.json'}")