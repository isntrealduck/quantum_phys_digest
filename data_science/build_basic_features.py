import hashlib
import json
from pathlib import Path


def word_count(text):
    """Return None when text is unavailable."""
    return len(text.split()) if text else None


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
    raise ValueError("The dataset changed after splitting.")


# Associate every paper with exactly one split.
split_by_id = {}

for split_name, paper_ids in splits["ids_by_split"].items():
    for paper_id in paper_ids:
        if paper_id in split_by_id:
            raise ValueError("A paper appears in multiple splits.")

        split_by_id[paper_id] = split_name

dataset_ids = [paper["openalex_id"] for paper in papers]

if len(dataset_ids) != len(set(dataset_ids)):
    raise ValueError("Duplicate paper IDs.")

if set(dataset_ids) != set(split_by_id):
    raise ValueError("The splits do not match the dataset.")


feature_rows = []

for paper in papers:
    authors = paper.get("authors") or []
    institution_ids = set()
    country_codes = set()
    authors_without_institutions = 0

    for author in authors:
        institutions = author.get("institutions") or []

        if not institutions:
            authors_without_institutions += 1

        for institution in institutions:
            if institution.get("id"):
                institution_ids.add(institution["id"])

            if institution.get("country_code"):
                country_codes.add(institution["country_code"])

    features = {
        "recorded_author_count": len(authors),
        "recorded_institution_count": len(institution_ids),
        "recorded_country_count": len(country_codes),
        "title_word_count": word_count(paper.get("title")),
        "abstract_word_count": word_count(paper.get("abstract")),
        "title_missing": int(not paper.get("title")),
        "abstract_missing": int(not paper.get("abstract")),
        "authorships_missing": int(not authors),
        "institution_ids_missing": int(not institution_ids),
        "country_codes_missing": int(not country_codes),
        "authors_without_institutions_fraction": (
            authors_without_institutions / len(authors)
            if authors else None
        ),
    }

    target = paper["five_year_citation_count"]

    if type(target) is not int or target < 0:
        raise ValueError("Invalid five-year citation target.")

    feature_rows.append({
        "openalex_id": paper["openalex_id"],
        "split": split_by_id[paper["openalex_id"]],
        "features": features,
        "target": target,
    })


# Save inputs separately from metadata used to identify each row.
feature_names = list(feature_rows[0]["features"])

output_folder = dataset_folder / "basic_features"
output_folder.mkdir(parents=True, exist_ok=True)

output = {
    "dataset_sha256": dataset_hash,
    "feature_names": feature_names,
    "target_name": "five_year_citation_count",
    "publication_time_features_verified": False,
    "rows": feature_rows,
}

(output_folder / "features.json").write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)


# Describe missing feature values using training rows only.
training_rows = [
    row for row in feature_rows if row["split"] == "train"
]

report = {
    "records": len(feature_rows),
    "records_by_split": {
        split_name: sum(
            row["split"] == split_name for row in feature_rows
        )
        for split_name in splits["ids_by_split"]
    },
    "feature_count": len(feature_names),
    "feature_names": feature_names,
    "training_null_counts": {
        name: sum(
            row["features"][name] is None
            for row in training_rows
        )
        for name in feature_names
    },
}

(output_folder / "report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))
print(f"\nFeature table: {output_folder / 'features.json'}")