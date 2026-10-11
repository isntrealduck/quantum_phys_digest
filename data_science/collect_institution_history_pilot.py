import getpass
import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TOPIC_IDS = ["T10020", "T10622", "T10682", "T11414", "T11804"]
MAX_PAGES = 5
PER_PAGE = 100

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


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# Load the dataset and verify that it matches the saved splits.
dataset_bytes = (dataset_folder / "papers_labeled.json").read_bytes()
papers = json.loads(dataset_bytes)

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()

if dataset_hash != splits["dataset_sha256"]:
    raise ValueError("Dataset differs from the split definition.")

train_ids = set(splits["ids_by_split"]["train"])

train = sorted(
    (paper for paper in papers if paper["openalex_id"] in train_ids),
    key=lambda paper: (
        paper["publication_date"],
        paper["openalex_id"],
    ),
)

if (
    len(train) != len(train_ids)
    or {paper["openalex_id"] for paper in train} != train_ids
):
    raise ValueError("Training records are missing or duplicated.")


# Select five distinct institutions using training metadata only.
pilot = []
seen_institutions = set()

for paper in train:
    for author in paper.get("authors") or []:
        for institution in author.get("institutions") or []:
            institution_id = institution.get("id")

            if not institution_id or institution_id in seen_institutions:
                continue

            short_id = institution_id.rsplit("/", 1)[-1]

            if not re.fullmatch(r"I\d+", short_id):
                raise ValueError("Invalid institution ID.")

            seen_institutions.add(institution_id)

            pilot.append({
                "institution_id": institution_id,
                "name": institution.get("name"),
                "example_paper_id": paper["openalex_id"],
                "cutoff_exclusive": paper["publication_date"],
            })

            if len(pilot) == 5:
                break

        if len(pilot) == 5:
            break

    if len(pilot) == 5:
        break

if len(pilot) != 5:
    raise ValueError("Fewer than five institutions were available.")


api_key = getpass.getpass("Paste your OpenAlex API key: ").strip()

if not api_key:
    raise SystemExit("No API key was entered.")

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

raw_folder = (
    project_folder / "data" / "raw" / "institution_history_pilot" / run_id
)
output_folder = dataset_folder / "institution_history_pilot" / run_id

raw_folder.mkdir(parents=True, exist_ok=False)
output_folder.mkdir(parents=True, exist_ok=False)

manifest = {
    "run_id": run_id,
    "dataset_sha256": dataset_hash,
    "started_at_utc": utc_now(),
    "status": "running",
    "corpus": "core",
    "topic_ids": TOPIC_IDS,
    "institution_scope": "Direct recorded affiliation",
    "maximum_pages_per_institution": MAX_PAGES,
    "selected_institutions": pilot,
    "requests": [],
}

save_json(raw_folder / "manifest.json", manifest)
report = []


try:
    for number, institution in enumerate(pilot, start=1):
        institution_id = institution["institution_id"]
        short_id = institution_id.rsplit("/", 1)[-1]
        cutoff = date.fromisoformat(institution["cutoff_exclusive"])
        last_date = cutoff - timedelta(days=1)

        print(f"\n[{number}/5] {institution['name']} ({short_id})")
        print(f"Collecting quantum articles dated before {cutoff}.")

        institution_raw = raw_folder / short_id
        institution_raw.mkdir()

        cursor = "*"
        used_cursors = set()
        works = {}
        expected_count = None

        for page_number in range(1, MAX_PAGES + 1):
            if cursor in used_cursors:
                raise ValueError("The API repeated a pagination cursor.")

            used_cursors.add(cursor)

            parameters = {
                "filter": (
                    f"institutions.id:{short_id},"
                    f"primary_topic.id:{'|'.join(TOPIC_IDS)},"
                    f"to_publication_date:{last_date},"
                    "type:article"
                ),
                "corpus": "core",
                "per_page": PER_PAGE,
                "cursor": cursor,
                "select": (
                    "id,display_name,publication_date,type,"
                    "primary_topic,authorships"
                ),
            }

            request = Request(
                "https://api.openalex.org/works?" + urlencode(parameters),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "quantum_phys_digest/0.6",
                },
            )

            time.sleep(1)

            with urlopen(request, timeout=60) as response:
                raw_bytes = response.read()

            raw_file = institution_raw / f"page_{page_number:04d}.json"
            raw_file.write_bytes(raw_bytes)

            manifest["requests"].append({
                "institution_id": institution_id,
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
                work_id = work["id"]
                publication_date = date.fromisoformat(
                    work["publication_date"]
                )
                topic_id = (
                    (work.get("primary_topic") or {}).get("id") or ""
                ).rsplit("/", 1)[-1]

                affiliation_ids = {
                    item.get("id")
                    for authorship in work.get("authorships") or []
                    for item in authorship.get("institutions") or []
                }

                if publication_date >= cutoff:
                    raise ValueError("A publication is outside the window.")

                if work.get("type") != "article":
                    raise ValueError("Unexpected publication type.")

                if topic_id not in TOPIC_IDS:
                    raise ValueError("Unexpected primary topic.")

                if institution_id not in affiliation_ids:
                    raise ValueError("Returned affiliation does not match.")

                if work_id in works:
                    raise ValueError("Duplicate publication ID.")

                works[work_id] = {
                    "openalex_id": work_id,
                    "title": work.get("display_name"),
                    "publication_date": work["publication_date"],
                    "primary_topic_id": topic_id,
                }

            print(
                f"  Page {page_number}: {len(works)} collected; "
                f"API reports {expected_count} total"
            )

            if len(works) > expected_count:
                raise ValueError("Collected more records than the API count.")

            # Stop once all reported records have been collected.
            if len(works) == expected_count:
                break

            next_cursor = data["meta"].get("next_cursor")

            if not results or next_cursor is None:
                raise ValueError("Pagination ended before the reported count.")

            cursor = next_cursor

        complete = len(works) == expected_count
        history_file = f"{short_id}_works.json"

        save_json(
            output_folder / history_file,
            sorted(
                works.values(),
                key=lambda work: (
                    work["publication_date"],
                    work["openalex_id"],
                ),
            ),
        )

        report.append({
            **institution,
            "api_reported_prior_quantum_article_count": expected_count,
            "articles_downloaded": len(works),
            "history_complete": complete,
            "validation_status": (
                "complete_count_verified" if complete
                else "downloaded_records_verified_history_capped"
            ),
            "history_file": history_file,
        })

        save_json(output_folder / "report.json", report)
        print(f"  Saved. Complete history: {complete}")

except (
    HTTPError, URLError, ValueError, KeyError, TypeError, OSError,
    KeyboardInterrupt,
) as error:
    manifest["status"] = "failed"
    manifest["error_type"] = type(error).__name__
    save_json(raw_folder / "manifest.json", manifest)

    if isinstance(error, HTTPError):
        message = f"OpenAlex returned HTTP {error.code}."
    elif isinstance(error, URLError):
        message = "A network connection failed."
    elif isinstance(error, KeyboardInterrupt):
        message = "Collection interrupted."
    else:
        message = str(error)

    raise SystemExit(
        f"{message}\nPreviously saved results remain in {output_folder}"
    ) from None


manifest["status"] = "completed"
manifest["completed_at_utc"] = utc_now()
save_json(raw_folder / "manifest.json", manifest)

print("\nInstitution pilot completed.")
print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\nResults: {output_folder / 'report.json'}")