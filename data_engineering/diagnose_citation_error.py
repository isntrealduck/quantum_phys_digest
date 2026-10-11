import json
from collections import Counter
from pathlib import Path


project_folder = Path(__file__).resolve().parent.parent
target_id = "https://openalex.org/W1870671750"

target_folder = (
    project_folder
    / "data"
    / "raw"
    / "openalex_five_year_citations"
    / "20261010T202611867850Z"
    / "20261010T204724059047Z"
    / "W1870671750"
)

# Inspect the latest attempt without changing any saved files.
attempt_folder = sorted(
    folder for folder in target_folder.iterdir() if folder.is_dir()
)[-1]

records = []
page_counts = []

for page_file in sorted(attempt_folder.glob("page_*.json")):
    response = json.loads(page_file.read_text(encoding="utf-8"))

    page_counts.append({
        "file": page_file.name,
        "api_reported_count": response["meta"]["count"],
        "next_cursor": response["meta"].get("next_cursor"),
        "records_returned": len(response["results"]),
    })

    for work in response["results"]:
        records.append({
            "id": work["id"],
            "title": work.get("display_name"),
            "publication_date": work.get("publication_date"),
            "page": page_file.name,
        })

id_counts = Counter(record["id"] for record in records)

report = {
    "target_id": target_id,
    "pages_saved": page_counts,
    "downloaded_records": len(records),
    "unique_downloaded_ids": len(id_counts),
    "self_reference_records": [
        record for record in records if record["id"] == target_id
    ],
    "duplicate_records": [
        record for record in records if id_counts[record["id"]] > 1
    ],
}

print(json.dumps(report, indent=2, ensure_ascii=False))