import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python inspect_openalex_collection.py COLLECTION_FOLDER"
    )

run_id = sys.argv[1]

if Path(run_id).name != run_id or run_id in {".", ".."}:
    raise ValueError("Supply a collection folder name, not a full path.")

project_folder = Path(__file__).resolve().parent
input_folder = (
    project_folder / "data" / "raw" / "openalex_historical" / run_id
)

manifest = json.loads(
    (input_folder / "manifest.json").read_text(encoding="utf-8")
)

if manifest["status"] != "completed":
    raise SystemExit("This collection did not finish successfully.")

papers = json.loads(
    (input_folder / "works.json").read_text(encoding="utf-8")
)


def has_institution(paper):
    """Check whether any author has an institution recorded."""
    return any(
        authorship.get("institutions")
        for authorship in paper.get("authorships") or []
    )


def valid_publication_date(paper):
    """Check the date format and agreement with publication year."""
    try:
        publication_date = date.fromisoformat(
            paper.get("publication_date") or ""
        )
        return publication_date.year == paper.get("publication_year")
    except (ValueError, TypeError):
        return False


# Count records by year and primary topic.
year_counts = Counter(
    str(paper.get("publication_year")) for paper in papers
)

topic_counts = Counter(
    (paper.get("primary_topic") or {}).get("display_name") or "Missing"
    for paper in papers
)

# Check whether multiple OpenAlex records share a DOI.
doi_to_ids = {}

for paper in papers:
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        doi_to_ids.setdefault(doi, []).append(paper["id"])

duplicate_dois = {
    doi: ids
    for doi, ids in doi_to_ids.items()
    if len(ids) > 1
}

report = {
    "collection_run": run_id,
    "records": len(papers),
    "unique_openalex_ids": len({paper["id"] for paper in papers}),
    "records_by_year": dict(sorted(year_counts.items())),
    "records_by_primary_topic": dict(sorted(topic_counts.items())),
    "missing_title": sum(
        not (paper.get("title") or "").strip() for paper in papers
    ),
    "missing_abstract": sum(
        not paper.get("abstract_inverted_index") for paper in papers
    ),
    "missing_authorships": sum(
        not paper.get("authorships") for paper in papers
    ),
    "without_recorded_institutions": sum(
        not has_institution(paper) for paper in papers
    ),
    "missing_doi": sum(
        not paper.get("doi") for paper in papers
    ),
    "invalid_or_inconsistent_publication_date": sum(
        not valid_publication_date(paper) for paper in papers
    ),
    "missing_current_citation_count": sum(
        paper.get("cited_by_count") is None for paper in papers
    ),
    "zero_current_citations": sum(
        paper.get("cited_by_count") == 0 for paper in papers
    ),
    "truncated_authorships": sum(
        paper.get("is_authors_truncated") is True for paper in papers
    ),
    "marked_retracted": sum(
        paper.get("is_retracted") is True for paper in papers
    ),
    "duplicate_dois": duplicate_dois,
}

# Save the report separately from the original downloaded records.
output_folder = (
    project_folder / "data" / "processed" / "openalex_historical" / run_id
)
output_folder.mkdir(parents=True, exist_ok=True)

report_file = output_folder / "collection_report.json"
report_file.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\nReport saved: {report_file}")