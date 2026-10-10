import json
import sys
from collections import Counter
from pathlib import Path

# Read the collection folder name from the command.
if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python check_matches.py COLLECTION_FOLDER_NAME"
    )

project_folder = Path(__file__).resolve().parent
collection_folder = (
    project_folder / "data" / "processed" / sys.argv[1] / "openalex"
)

# Find the most recent matching run.
runs = sorted(
    folder
    for folder in collection_folder.iterdir()
    if folder.is_dir() and (folder / "matches.json").exists()
)

if not runs:
    raise SystemExit("No saved matching results were found.")

latest_run = runs[-1]
matches = json.loads(
    (latest_run / "matches.json").read_text(encoding="utf-8")
)

# Count how often each matching status appears.
status_counts = Counter(row["status"] for row in matches)

print(f"Matching run: {latest_run.name}")
print(f"Saved lookups: {len(matches)}")

for status, count in sorted(status_counts.items()):
    print(f"{status}: {count}")

# List records that need investigation.
problems = [
    row for row in matches
    if row["status"] != "doi_candidate_pending_review"
]

print(f"\nRecords needing investigation: {len(problems)}")

for row in problems:
    print(f"{row['arxiv_id']} | {row['status']} | {row['input_doi']}")

print(f"\nReview file: {latest_run / 'review.csv'}")

# Inspect the candidates for records needing investigation.
for row in problems:
    raw_file = project_folder / row["raw_response_file"]
    response = json.loads(raw_file.read_text(encoding="utf-8"))

    print(f"\n--- arXiv {row['arxiv_id']} ---")
    print(f"Original title: {row['arxiv_title']}")
    print(f"Original authors: {row['arxiv_authors']}")
    print(f"Original DOI: {row['input_doi']}")
    print(f"Reported candidate count: {response['meta']['count']}")
    print(f"Candidates saved: {len(response['results'])}")

    for number, candidate in enumerate(response["results"], start=1):
        authors = [
            (authorship.get("author") or {}).get("display_name") or ""
            for authorship in candidate.get("authorships") or []
        ]

        print(f"\nCandidate {number}:")
        print(f"  ID: {candidate.get('id')}")
        print(f"  Title: {candidate.get('display_name')}")
        print(f"  Authors: {'; '.join(authors)}")
        print(f"  DOI: {candidate.get('doi')}")
        print(f"  Publication date: {candidate.get('publication_date')}")