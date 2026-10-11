import getpass
import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


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

AUTHOR_HISTORY_RUN = "20261011T101956741223Z"
MAX_PAGES_PER_ARTICLE = 5


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# Verify that the existing histories belong to the current queue.
queue_bytes = (
    dataset_folder / "author_history_queue.json"
).read_bytes()

queue = json.loads(queue_bytes)
queue_hash = hashlib.sha256(queue_bytes).hexdigest()

dataset_hash = hashlib.sha256(
    (dataset_folder / "papers_labeled.json").read_bytes()
).hexdigest()

if dataset_hash != queue["dataset_sha256"]:
    raise ValueError("Dataset differs from the author queue.")

jobs_by_id = {job["author_id"]: job for job in queue["jobs"]}

history_folder = (
    dataset_folder / "author_histories" / AUTHOR_HISTORY_RUN
)

# Select a small completed history; never read an unfinished checkpoint.
history = None

for history_file in sorted(history_folder.glob("A*_history.json")):
    candidate = json.loads(
        history_file.read_text(encoding="utf-8")
    )

    if (
        candidate.get("validation_status") == "passed"
        and 1 <= len(candidate.get("works", [])) <= 5
    ):
        history = candidate
        break

if history is None:
    raise SystemExit("No completed history with 1–5 articles was found.")

author_id = history["author_id"]
job = jobs_by_id[author_id]
cutoff = date.fromisoformat(history["cutoff_exclusive"])
last_date = cutoff - timedelta(days=1)

prior_works = history["works"]
prior_ids = [work["openalex_id"] for work in prior_works]

if (
    history["queue_sha256"] != queue_hash
    or history["cutoff_exclusive"] != job["cutoff_exclusive"]
    or len(prior_ids) != len(set(prior_ids))
    or len(prior_works) != history["api_reported_count"]
):
    raise ValueError("The selected author history failed validation.")

for work in prior_works:
    if date.fromisoformat(work["publication_date"]) >= cutoff:
        raise ValueError("A prior article is outside its history window.")


print(f"Author: {job['example_name']} ({author_id})")
print(f"Earlier articles: {len(prior_works)}")
print(f"Counting citations dated before {cutoff}.")

api_key = getpass.getpass("Paste your OpenAlex API key: ").strip()

if not api_key:
    raise SystemExit("No API key was entered.")

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

raw_folder = (
    project_folder / "data" / "raw" / "author_citation_pilot" / run_id
)
output_folder = dataset_folder / "author_citation_pilot" / run_id

raw_folder.mkdir(parents=True, exist_ok=False)
output_folder.mkdir(parents=True, exist_ok=False)

report = {
    "run_id": run_id,
    "author_history_run": AUTHOR_HISTORY_RUN,
    "dataset_sha256": dataset_hash,
    "author_id": author_id,
    "author_name": job["example_name"],
    "cutoff_exclusive": cutoff.isoformat(),
    "corpus": "core",
    "prior_work_scope": "Recorded articles only",
    "citation_window": (
        "prior_article_date <= citing_article_date < cutoff"
    ),
    "status": "running",
    "started_at_utc": utc_now(),
    "requests_completed": 0,
    "articles": [],
}

save_json(output_folder / "report.json", report)


try:
    for number, prior_work in enumerate(prior_works, start=1):
        target_id = prior_work["openalex_id"]
        short_id = target_id.rsplit("/", 1)[-1]
        start = date.fromisoformat(prior_work["publication_date"])

        print(f"\n[{number}/{len(prior_works)}] {prior_work['title']}")

        article_raw = raw_folder / short_id
        article_raw.mkdir()

        cursor = "*"
        used_cursors = set()
        returned_works = {}
        expected_count = None

        for page_number in range(1, MAX_PAGES_PER_ARTICLE + 1):
            if cursor in used_cursors:
                raise ValueError("The API repeated a cursor.")

            used_cursors.add(cursor)

            parameters = {
                "filter": (
                    f"cites:{short_id},"
                    f"from_publication_date:{start},"
                    f"to_publication_date:{last_date}"
                ),
                "corpus": "core",
                "per_page": 100,
                "cursor": cursor,
                "select": (
                    "id,display_name,publication_date,referenced_works"
                ),
            }

            save_json(
                article_raw / f"request_{page_number:04d}.json",
                {
                    "parameters": parameters,
                    "requested_at_utc": utc_now(),
                },
            )

            request = Request(
                "https://api.openalex.org/works?"
                + urlencode(parameters),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "quantum_phys_digest/0.7",
                },
            )

            # Retry temporary connection and server failures.
            for attempt in range(1, 6):
                time.sleep(1)

                try:
                    with urlopen(request, timeout=60) as response:
                        raw_bytes = response.read()
                    break
                except HTTPError as error:
                    if error.code not in {500, 502, 503, 504}:
                        raise
                    if attempt == 5:
                        raise
                    time.sleep(min(5 * 2 ** (attempt - 1), 40))
                except (URLError, TimeoutError, ConnectionError):
                    if attempt == 5:
                        raise
                    time.sleep(min(5 * 2 ** (attempt - 1), 40))

            (article_raw / f"page_{page_number:04d}.json").write_bytes(
                raw_bytes
            )

            report["requests_completed"] += 1
            data = json.loads(raw_bytes)
            page_count = data["meta"]["count"]

            if expected_count is None:
                expected_count = page_count
            elif expected_count != page_count:
                raise ValueError("API count changed during pagination.")

            results = data["results"]

            for work in results:
                work_id = work["id"]
                citing_date = date.fromisoformat(
                    work["publication_date"]
                )

                if not start <= citing_date < cutoff:
                    raise ValueError("Citation is outside its date window.")

                if target_id not in (work.get("referenced_works") or []):
                    raise ValueError("Returned work does not cite the article.")

                if work_id in returned_works:
                    raise ValueError("Duplicate citing work ID.")

                returned_works[work_id] = {
                    "openalex_id": work_id,
                    "title": work.get("display_name"),
                    "publication_date": work["publication_date"],
                }

            print(
                f"  Page {page_number}: {len(returned_works)} "
                f"of {expected_count} returned records"
            )

            if len(returned_works) > expected_count:
                raise ValueError("Collected more records than reported.")

            if len(returned_works) == expected_count:
                break

            next_cursor = data["meta"].get("next_cursor")

            if not results or next_cursor is None:
                raise ValueError("Pagination ended before the reported count.")

            cursor = next_cursor

        complete = len(returned_works) == expected_count

        # Exclude erroneous records in which an article cites itself.
        excluded_self_reference_count = int(
            target_id in returned_works
        )
        returned_works.pop(target_id, None)

        evidence_file = f"{short_id}_citing_works.json"

        save_json(
            output_folder / evidence_file,
            sorted(
                returned_works.values(),
                key=lambda work: (
                    work["publication_date"],
                    work["openalex_id"],
                ),
            ),
        )

        report["articles"].append({
            "openalex_id": target_id,
            "title": prior_work["title"],
            "publication_date": start.isoformat(),
            "api_reported_count": expected_count,
            "excluded_self_reference_count": excluded_self_reference_count,
            "history_complete": complete,
            "historical_citations": (
                len(returned_works) if complete else None
            ),
            "downloaded_valid_citing_works": len(returned_works),
            "citing_works_file": evidence_file,
        })

        save_json(output_folder / "report.json", report)


except (
    HTTPError, URLError, ValueError, KeyError, TypeError,
    OSError, KeyboardInterrupt,
) as error:
    report["status"] = "failed"
    report["error_type"] = type(error).__name__
    save_json(output_folder / "report.json", report)

    if isinstance(error, HTTPError):
        message = f"OpenAlex returned HTTP {error.code}."
    elif isinstance(error, URLError):
        message = "A network connection failed."
    elif isinstance(error, KeyboardInterrupt):
        message = "Pilot interrupted."
    else:
        message = str(error)

    raise SystemExit(
        f"{message}\nSaved progress: {output_folder}"
    ) from None


all_complete = all(
    article["history_complete"] for article in report["articles"]
)

report["status"] = "completed"
report["completed_at_utc"] = utc_now()
report["all_article_histories_complete"] = all_complete

# Sum citation links: one citing paper can cite several earlier articles.
report["author_prior_article_citation_total"] = (
    sum(article["historical_citations"] for article in report["articles"])
    if all_complete else None
)

save_json(output_folder / "report.json", report)

print("\nAuthor citation pilot completed.")
print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\nReport: {output_folder / 'report.json'}")