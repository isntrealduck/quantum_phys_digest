import csv
import getpass
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# Select the historical collection to use.
if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python match_openalex_sample.py COLLECTION_FOLDER_NAME"
    )

run_id = sys.argv[1]
if Path(run_id).name != run_id or run_id in {".", ".."}:
    raise ValueError("Supply a collection folder name, not a full path.")

project_folder = Path(__file__).resolve().parent
input_file = project_folder / "data" / "processed" / run_id / "papers.json"

papers = json.loads(input_file.read_text(encoding="utf-8"))

# Select two DOI-bearing papers from each year.
sample = []
for year in range(2015, 2020):
    eligible = [
        paper for paper in papers
        if paper["doi"] and paper["submitted_at"].startswith(str(year))
    ]
    if len(eligible) < 2:
        raise ValueError(f"Fewer than two papers have DOIs in {year}.")
    sample.extend(eligible[:2])

# Read the key without displaying it or saving it to a file.
api_key = getpass.getpass("Paste your OpenAlex API key and press Enter: ").strip()
if not api_key:
    raise ValueError("No API key was entered.")

# Preserve each matching run separately.
match_run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
raw_folder = project_folder / "data" / "raw" / "openalex" / run_id / match_run
output_folder = project_folder / "data" / "processed" / run_id / "openalex" / match_run
raw_folder.mkdir(parents=True, exist_ok=False)
output_folder.mkdir(parents=True, exist_ok=False)


def normalize_doi(value):
    """Remove a DOI URL prefix and normalize letter case."""
    value = (value or "").strip()
    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I
    ).lower()


rows = []

for number, paper in enumerate(sample, start=1):
    time.sleep(1)

    parameters = {
        "filter": f"doi:{paper['doi']}",
        "per_page": 2,
    }
    url = f"https://api.openalex.org/works?{urlencode(parameters)}"

    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "quantum_phys_digest/0.1",
        },
    )

    print(f"[{number}/{len(sample)}] Looking up {paper['arxiv_id']}...")

    try:
        with urlopen(request, timeout=60) as response:
            raw_json = response.read()
    except HTTPError as error:
        raise SystemExit(
            f"OpenAlex returned HTTP {error.code}. "
            "401/403: check authentication. "
            "429: rate limit or daily budget reached. "
            "Previously saved results remain available."
        ) from None

    retrieved_at = datetime.now(timezone.utc).isoformat()
    raw_file = raw_folder / f"{paper['arxiv_id']}.json"
    raw_file.write_bytes(raw_json)

    data = json.loads(raw_json)
    candidates = data["results"]
    candidate_count = data["meta"]["count"]

    # Only use a single candidate; ambiguous results require review.
    work = candidates[0] if candidate_count == 1 and len(candidates) == 1 else None

    if candidate_count == 0:
        status = "not_found_by_doi"
    elif work is None:
        status = "ambiguous"
    elif normalize_doi(work.get("doi")) != normalize_doi(paper["doi"]):
        status = "returned_doi_mismatch"
    else:
        status = "doi_candidate_pending_review"

    openalex_authors = []
    if work is not None:
        for authorship in work.get("authorships") or []:
            author = authorship.get("author") or {}
            if author.get("display_name"):
                openalex_authors.append(author["display_name"])

    row = {
        "arxiv_id": paper["arxiv_id"],
        "input_doi": paper["doi"],
        "status": status,
        "candidate_count": candidate_count,
        "arxiv_title": paper["title"],
        "openalex_title": work.get("display_name") if work else None,
        "arxiv_authors": "; ".join(a["name"] for a in paper["authors"]),
        "openalex_authors": "; ".join(openalex_authors),
        "arxiv_submitted_at": paper["submitted_at"],
        "openalex_publication_date": work.get("publication_date") if work else None,
        "openalex_id": work.get("id") if work else None,
        "returned_doi": work.get("doi") if work else None,
        "current_citation_count": work.get("cited_by_count") if work else None,
        "retrieved_at_utc": retrieved_at,
        "raw_response_file": str(raw_file.relative_to(project_folder)),
        "manual_review": "pending",
    }
    rows.append(row)

    # Save progress after every lookup.
    (output_folder / "matches.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (output_folder / "review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Result: {status}")

print(f"\nCompleted {len(rows)} DOI lookups.")
print(f"Review table: {output_folder / 'review.csv'}")