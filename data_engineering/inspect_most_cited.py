import json
from pathlib import Path


project_folder = Path(__file__).resolve().parent.parent

dataset_file = (
    project_folder
    / "data"
    / "processed"
    / "openalex_historical"
    / "20261010T184050630942Z"
    / "labeled"
    / "20261010T195747585977Z"
    / "papers_labeled.json"
)

papers = json.loads(dataset_file.read_text(encoding="utf-8"))

paper = max(
    papers,
    key=lambda record: record["five_year_citation_count"],
)

print("OpenAlex ID:", paper["openalex_id"])
print("DOI:", paper.get("doi"))
print("Title:", paper["title"])
print("Publication date:", paper["publication_date"])
print("Type:", paper.get("type"))
print("Topic:", paper.get("observed_primary_topic"))
print("Five-year citations:", paper["five_year_citation_count"])
print("Current citations at collection:", paper["current_citation_count"])
print("Target provenance:", paper["target_provenance"])

citations_file = (
    project_folder
    / paper["target_provenance"]["citing_works_file"]
)

citing_works = json.loads(
    citations_file.read_text(encoding="utf-8")
)

ordered = sorted(
    citing_works,
    key=lambda work: (
        work["publication_date"],
        work["openalex_id"],
    ),
)

counts_by_year = {}

for work in ordered:
    year = work["publication_date"][:4]
    counts_by_year[year] = counts_by_year.get(year, 0) + 1

print("\nCiting works by publication year:")
print(json.dumps(counts_by_year, indent=2))

print("\nThree earliest citing works:")
print(json.dumps(ordered[:3], indent=2, ensure_ascii=False))

print("\nThree latest citing works:")
print(json.dumps(ordered[-3:], indent=2, ensure_ascii=False))