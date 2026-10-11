import getpass
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def fifth_anniversary(start):
    """Use February 28 if a February 29 anniversary is unavailable."""
    try:
        return start.replace(year=start.year + 5)
    except ValueError:
        return start.replace(year=start.year + 5, day=28)


# Select the normalized historical collection.
if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python collect_five_year_citations.py COLLECTION_FOLDER_NAME"
    )

collection_run = sys.argv[1]

if Path(collection_run).name != collection_run or collection_run in {".", ".."}:
    raise ValueError("Supply a collection folder name, not a full path.")

project_folder = Path(__file__).resolve().parent.parent
collection_folder = (
    project_folder
    / "data"
    / "processed"
    / "openalex_historical"
    / collection_run
)

papers = json.loads(
    (collection_folder / "papers.json").read_text(encoding="utf-8")
)


# Process every paper in the normalized collection.
pilot = sorted(
    papers,
    key=lambda paper: (
        paper["publication_date"],
        paper["openalex_id"],
    ),
)

if not pilot:
    raise ValueError("The normalized collection is empty.")

ids = [paper["openalex_id"] for paper in pilot]

if len(ids) != len(set(ids)):
    raise ValueError("The normalized collection contains duplicate IDs.")


api_key = getpass.getpass("Paste your OpenAlex API key: ").strip()

if not api_key:
    raise ValueError("No API key was entered.")


# Preserve each pilot run separately.
pilot_run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

raw_folder = (
    project_folder
    / "data"
    / "raw"
    / "openalex_five_year_citations"
    / collection_run
    / pilot_run
)

output_folder = collection_folder / "citation_labels" / pilot_run

raw_folder.mkdir(parents=True, exist_ok=False)
output_folder.mkdir(parents=True, exist_ok=False)

manifest = {
    "collection_run": collection_run,
    "pilot_run": pilot_run,
    "started_at_utc": utc_now(),
    "status": "running",
    "corpus": "core",
    "window_definition": (
        "publication_date <= citing_publication_date < fifth_anniversary"
    ),
    "february_29_rule": "Use February 28 for the fifth anniversary.",
    "selection_rule": "All papers in the normalized historical collection.",
    "selected_openalex_ids": [paper["openalex_id"] for paper in pilot],
    "requests": [],
}

save_json(raw_folder / "manifest.json", manifest)
save_json(output_folder / "selected_papers.json", pilot)

labels = []

try:
    for number, paper in enumerate(pilot, start=1):
        target_id = paper["openalex_id"]
        short_id = target_id.rsplit("/", 1)[-1]

        if not re.fullmatch(r"W\d+", short_id):
            raise ValueError("Invalid OpenAlex work ID.")

        start = date.fromisoformat(paper["publication_date"])
        end_exclusive = fifth_anniversary(start)
        last_included_date = end_exclusive - timedelta(days=1)

        if end_exclusive > datetime.now(timezone.utc).date():
            raise ValueError("This paper has not completed its five-year window.")

        print(f"\n[{number}/{len(pilot)}] {short_id}")
        print(f"Title: {paper['title']}")
        print(f"Window: {start} to {last_included_date}, inclusive")

        paper_raw_folder = raw_folder / short_id
        paper_raw_folder.mkdir()

        cursor = "*"
        used_cursors = set()
        citing_works = {}
        expected_count = None
        page_number = 0

        while True:
            if cursor in used_cursors:
                raise ValueError("The API repeated a pagination cursor.")

            used_cursors.add(cursor)
            page_number += 1

            parameters = {
                "filter": (
                    f"cites:{short_id},"
                    f"from_publication_date:{start},"
                    f"to_publication_date:{last_included_date}"
                ),
                "corpus": "core",
                "per_page": 100,
                "cursor": cursor,
                "select": "id,display_name,publication_date,referenced_works",
            }

            url = (
                "https://api.openalex.org/works?"
                + urlencode(parameters)
            )

            request = Request(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "quantum_phys_digest/0.3",
                },
            )

            time.sleep(1)

            with urlopen(request, timeout=60) as response:
                raw_bytes = response.read()

            raw_file = paper_raw_folder / f"page_{page_number:04d}.json"
            raw_file.write_bytes(raw_bytes)

            manifest["requests"].append({
                "target_openalex_id": target_id,
                "parameters": parameters,
                "retrieved_at_utc": utc_now(),
                "raw_response_file": raw_file.relative_to(
                    project_folder
                ).as_posix(),
            })
            save_json(raw_folder / "manifest.json", manifest)

            data = json.loads(raw_bytes)
            page_count = data["meta"]["count"]

            if expected_count is None:
                expected_count = page_count
            elif page_count != expected_count:
                raise ValueError("The API count changed during pagination.")

            results = data["results"]

            for work in results:
                citing_id = work["id"]
                citing_date = date.fromisoformat(work["publication_date"])

                if not start <= citing_date < end_exclusive:
                    raise ValueError("A citing work is outside the window.")

                if target_id not in (work.get("referenced_works") or []):
                    raise ValueError("A returned work does not reference the target.")

                if citing_id == target_id:
                    raise ValueError("The target work references itself.")

                if citing_id in citing_works:
                    raise ValueError("A citing work ID appeared more than once.")

                citing_works[citing_id] = {
                    "openalex_id": citing_id,
                    "title": work.get("display_name"),
                    "publication_date": work["publication_date"],
                }

            print(
                f"  Page {page_number}: "
                f"{len(citing_works)} unique citing works collected"
            )

            next_cursor = data["meta"].get("next_cursor")

            if not results or next_cursor is None:
                break

            cursor = next_cursor

        if len(citing_works) != expected_count:
            raise ValueError("Collected IDs do not match the API's reported count.")

        citations_file = output_folder / f"{short_id}_citing_works.json"
        save_json(
            citations_file,
            sorted(
                citing_works.values(),
                key=lambda work: (
                    work["publication_date"],
                    work["openalex_id"],
                ),
            ),
        )

        labels.append({
            "openalex_id": target_id,
            "title": paper["title"],
            "publication_date": start.isoformat(),
            "window_end_exclusive": end_exclusive.isoformat(),
            "five_year_citation_count": len(citing_works),
            "historical_current_citation_count": paper["current_citation_count"],
            "api_reported_window_count": expected_count,
            "validation_status": "passed",
            "manual_review": "pending",
            "citing_works_file": citations_file.name,
            "completed_at_utc": utc_now(),
        })

        save_json(output_folder / "labels.json", labels)
        print(f"  Five-year citation count: {len(citing_works)}")

except (HTTPError, URLError, ValueError, KeyError, TypeError) as error:
    manifest["status"] = "failed"
    manifest["failed_at_utc"] = utc_now()
    manifest["error_type"] = type(error).__name__

    if isinstance(error, HTTPError):
        manifest["http_status"] = error.code
        message = f"OpenAlex returned HTTP {error.code}."
    elif isinstance(error, URLError):
        message = "A network connection failed."
    else:
        message = f"A validation failed: {error}"

    save_json(raw_folder / "manifest.json", manifest)

    raise SystemExit(
        f"{message}\n"
        "Previously saved responses and completed labels remain available."
    ) from None


manifest["status"] = "completed"
manifest["completed_at_utc"] = utc_now()
manifest["labels_created"] = len(labels)
save_json(raw_folder / "manifest.json", manifest)

print(f"\nPilot completed: {len(labels)} labels created.")
print(f"Results: {output_folder / 'labels.json'}")