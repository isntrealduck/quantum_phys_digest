import json
import statistics
import sys
from datetime import date
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fifth_anniversary(start):
    try:
        return start.replace(year=start.year + 5)
    except ValueError:
        return start.replace(year=start.year + 5, day=28)


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python inspect_citation_labels.py COLLECTION_FOLDER_NAME"
    )

collection_run = sys.argv[1]

if Path(collection_run).name != collection_run or collection_run in {".", ".."}:
    raise ValueError("Supply a collection folder name, not a full path.")

project_folder = Path(__file__).resolve().parent
collection_folder = (
    project_folder
    / "data"
    / "processed"
    / "openalex_historical"
    / collection_run
)

papers = read_json(collection_folder / "papers.json")

# Find the most recent completed citation collection.
completed_runs = []

for folder in (collection_folder / "citation_labels").iterdir():
    if not folder.is_dir():
        continue

    manifest_file = (
        project_folder
        / "data"
        / "raw"
        / "openalex_five_year_citations"
        / collection_run
        / folder.name
        / "manifest.json"
    )

    if manifest_file.exists():
        manifest = read_json(manifest_file)

        if manifest.get("status") == "completed":
            completed_runs.append(folder)

if not completed_runs:
    raise ValueError("No completed citation collection was found.")

labels_folder = sorted(completed_runs, key=lambda folder: folder.name)[-1]
labels = read_json(labels_folder / "labels.json")

paper_ids = [paper["openalex_id"] for paper in papers]
label_ids = [label["openalex_id"] for label in labels]

if len(paper_ids) != len(set(paper_ids)):
    raise ValueError("Duplicate IDs in papers.json.")

if len(label_ids) != len(set(label_ids)):
    raise ValueError("Duplicate IDs in labels.json.")

if set(paper_ids) != set(label_ids):
    raise ValueError("Paper IDs and label IDs do not match exactly.")

papers_by_id = {paper["openalex_id"]: paper for paper in papers}
counts = []

for label in labels:
    paper = papers_by_id[label["openalex_id"]]
    count = label["five_year_citation_count"]

    if type(count) is not int or count < 0:
        raise ValueError("A citation count is not a nonnegative integer.")

    if label["validation_status"] != "passed":
        raise ValueError("A label did not pass collection validation.")

    start = date.fromisoformat(paper["publication_date"])
    end = fifth_anniversary(start)

    if label["publication_date"] != start.isoformat():
        raise ValueError("A label's publication date differs from its paper.")

    if label["window_end_exclusive"] != end.isoformat():
        raise ValueError("An incorrect five-year window was recorded.")

    filename = label["citing_works_file"]

    if Path(filename).name != filename:
        raise ValueError("Expected a citing-works filename, not a path.")

    citing_works = read_json(labels_folder / filename)
    citing_ids = [work["openalex_id"] for work in citing_works]

    if len(citing_ids) != len(set(citing_ids)):
        raise ValueError("Duplicate citing IDs were found.")

    if len(citing_works) != count:
        raise ValueError("A label differs from its saved citing-work count.")

    if label["api_reported_window_count"] != count:
        raise ValueError("A label differs from its reported API count.")

    for work in citing_works:
        citing_date = date.fromisoformat(work["publication_date"])

        if not start <= citing_date < end:
            raise ValueError("A saved citing work is outside the window.")

    counts.append(count)

report = {
    "collection_run": collection_run,
    "citation_run": labels_folder.name,
    "papers": len(papers),
    "labels": len(labels),
    "unique_label_ids": len(set(label_ids)),
    "all_checks_passed": True,
    "zero_five_year_citations": sum(count == 0 for count in counts),
    "minimum": min(counts),
    "median": statistics.median(counts),
    "mean": round(statistics.mean(counts), 2),
    "maximum": max(counts),
}

report_file = labels_folder / "label_quality_report.json"
report_file.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))
print(f"\nReport saved: {report_file}")