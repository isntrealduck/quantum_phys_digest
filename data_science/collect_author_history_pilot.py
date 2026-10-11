import getpass
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def save_json(path, value):
    """Save progress without leaving a partially written JSON file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


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

inventory = json.loads(
    (dataset_folder / "historical_entity_inventory.json").read_text(
        encoding="utf-8"
    )
)

pilot_authors = inventory["pilot_authors"]

if not pilot_authors:
    raise SystemExit("The inventory contains no pilot authors.")

api_key = getpass.getpass("Paste your OpenAlex API key: ").strip()

if not api_key:
    raise SystemExit("No API key was entered.")

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

raw_folder = (
    project_folder / "data" / "raw" / "author_history_pilot" / run_id
)
output_folder = dataset_folder / "author_history_pilot" / run_id

raw_folder.mkdir(parents=True, exist_ok=False)
output_folder.mkdir(parents=True, exist_ok=False)

manifest = {
    "run_id": run_id,
    "status": "running",
    "corpus": "core",
    "work_type": "article",
    "date_rule": "Prior publication date < target publication date",
    "pilot_authors": pilot_authors,
    "requests": [],
}

summaries = []
save_json(raw_folder / "manifest.json", manifest)

try:
    for number, author in enumerate(pilot_authors, start=1):
        author_id = author["author_id"]
        short_id = author_id.rsplit("/", 1)[-1]

        if not re.fullmatch(r"A\d+", short_id):
            raise ValueError(f"Invalid author ID: {author_id}")

        target_date = date.fromisoformat(
            author["example_publication_date"]
        )
        last_included_date = target_date - timedelta(days=1)

        print(
            f"\n[{number}/{len(pilot_authors)}] "
            f"{author['name']} ({short_id})"
        )
        print(f"Collecting articles dated before {target_date}.")

        author_raw_folder = raw_folder / short_id
        author_raw_folder.mkdir()

        cursor = "*"
        used_cursors = set()
        works = {}
        expected_count = None
        page_number = 0

        while True:
            if cursor in used_cursors:
                raise ValueError("The API repeated a cursor.")

            used_cursors.add(cursor)
            page_number += 1

            parameters = {
                "filter": (
                    f"authorships.author.id:{short_id},"
                    f"to_publication_date:{last_included_date},"
                    "type:article"
                ),
                "corpus": "core",
                "per_page": 100,
                "cursor": cursor,
                "select": "id,display_name,publication_date,type",
            }

            url = (
                "https://api.openalex.org/works?"
                + urlencode(parameters)
            )
            request = Request(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "quantum_phys_digest/0.4",
                },
            )

            time.sleep(1)

            with urlopen(request, timeout=60) as response:
                raw_bytes = response.read()

            raw_file = (
                author_raw_folder / f"page_{page_number:04d}.json"
            )
            raw_file.write_bytes(raw_bytes)

            manifest["requests"].append({
                "author_id": author_id,
                "parameters": parameters,
                "retrieved_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "raw_response_file": raw_file.relative_to(
                    project_folder
                ).as_posix(),
            })
            save_json(raw_folder / "manifest.json", manifest)

            data = json.loads(raw_bytes)
            reported_count = data["meta"]["count"]

            if expected_count is None:
                expected_count = reported_count
            elif reported_count != expected_count:
                raise ValueError(
                    "The API count changed during pagination."
                )

            results = data["results"]

            for work in results:
                work_id = work["id"]
                publication_date = date.fromisoformat(
                    work["publication_date"]
                )

                if publication_date >= target_date:
                    raise ValueError(
                        "An article is outside the historical window."
                    )

                if work["type"] != "article":
                    raise ValueError("Unexpected work type.")

                if work_id in works:
                    raise ValueError("Duplicate article ID.")

                works[work_id] = {
                    "openalex_id": work_id,
                    "title": work.get("display_name"),
                    "publication_date": work["publication_date"],
                }

            print(
                f"  Page {page_number}: "
                f"{len(works)} earlier articles collected"
            )

            next_cursor = data["meta"].get("next_cursor")

            if not results or next_cursor is None:
                break

            cursor = next_cursor

        if len(works) != expected_count:
            raise ValueError(
                "Collected articles do not match the API count."
            )

        ordered_works = sorted(
            works.values(),
            key=lambda work: (
                work["publication_date"],
                work["openalex_id"],
            ),
        )

        history_file = output_folder / f"{short_id}_works.json"
        save_json(history_file, ordered_works)

        summaries.append({
            **author,
            "prior_article_count": len(ordered_works),
            "earliest_recorded_article_date": (
                ordered_works[0]["publication_date"]
                if ordered_works else None
            ),
            "history_file": history_file.name,
            "validation_status": "passed",
        })
        save_json(output_folder / "summary.json", summaries)

        print(f"  Earlier article count: {len(ordered_works)}")

except (HTTPError, URLError, ValueError, KeyError, TypeError) as error:
    manifest["status"] = "failed"
    manifest["error_type"] = type(error).__name__
    save_json(raw_folder / "manifest.json", manifest)

    if isinstance(error, HTTPError):
        message = f"OpenAlex returned HTTP {error.code}."
    elif isinstance(error, URLError):
        message = "Could not connect to OpenAlex."
    else:
        message = str(error)

    raise SystemExit(
        f"{message}\nSaved progress: {output_folder}"
    ) from None

manifest["status"] = "completed"
manifest["authors_completed"] = len(summaries)
save_json(raw_folder / "manifest.json", manifest)

print("\nPilot completed.")
print(json.dumps(summaries, indent=2, ensure_ascii=False))
print(f"\nResults: {output_folder / 'summary.json'}")