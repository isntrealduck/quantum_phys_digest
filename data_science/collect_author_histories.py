import argparse
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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


parser = argparse.ArgumentParser()
parser.add_argument("--resume", help="Existing author-history run ID")
parser.add_argument(
    "--limit",
    type=int,
    help="Maximum number of additional authors to complete",
)
args = parser.parse_args()

if args.limit is not None and args.limit < 1:
    parser.error("--limit must be at least 1.")

if args.resume and not re.fullmatch(r"\d{8}T\d{12}Z", args.resume):
    parser.error("Invalid run ID.")


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

queue_bytes = (
    dataset_folder / "author_history_queue.json"
).read_bytes()
queue_hash = hashlib.sha256(queue_bytes).hexdigest()
queue = json.loads(queue_bytes)

dataset_hash = hashlib.sha256(
    (dataset_folder / "papers_labeled.json").read_bytes()
).hexdigest()

if dataset_hash != queue["dataset_sha256"]:
    raise SystemExit("The dataset no longer matches the queue.")

jobs = queue["jobs"]
jobs_by_id = {job["author_id"]: job for job in jobs}

if not jobs or len(jobs_by_id) != len(jobs):
    raise SystemExit("The queue is empty or contains duplicate authors.")

for job in jobs:
    short_id = job["author_id"].rsplit("/", 1)[-1]

    if not re.fullmatch(r"A\d+", short_id):
        raise SystemExit("Invalid author ID in the queue.")

    date.fromisoformat(job["cutoff_exclusive"])


run_id = args.resume or timestamp()

raw_folder = (
    project_folder / "data" / "raw" / "author_histories" / run_id
)
output_folder = dataset_folder / "author_histories" / run_id
manifest_file = raw_folder / "manifest.json"

if args.resume:
    manifest = json.loads(
        manifest_file.read_text(encoding="utf-8")
    )

    if manifest["queue_sha256"] != queue_hash:
        raise SystemExit("The queue changed; cannot resume this run.")
else:
    raw_folder.mkdir(parents=True, exist_ok=False)
    output_folder.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "status": "running",
        "queue_sha256": queue_hash,
        "corpus": queue["corpus"],
        "work_type": queue["work_type"],
        "total_authors": len(jobs),
    }
    save_json(raw_folder / "queue.json", queue)
    save_json(manifest_file, manifest)


# Validate saved results before treating them as completed.
completed = set()

for result_file in sorted(output_folder.glob("A*_history.json")):
    result = json.loads(result_file.read_text(encoding="utf-8"))
    author_id = result["author_id"]

    if author_id not in jobs_by_id or author_id in completed:
        raise SystemExit("Unexpected or duplicate saved author.")

    job = jobs_by_id[author_id]
    works = result["works"]
    cutoff = date.fromisoformat(job["cutoff_exclusive"])
    work_ids = [work["openalex_id"] for work in works]

    if (
        result["queue_sha256"] != queue_hash
        or result["cutoff_exclusive"] != job["cutoff_exclusive"]
        or result["validation_status"] != "passed"
        or len(work_ids) != len(set(work_ids))
        or len(works) != result["api_reported_count"]
    ):
        raise SystemExit(f"Invalid saved result: {result_file}")

    for work in works:
        if date.fromisoformat(work["publication_date"]) >= cutoff:
            raise SystemExit("Saved article is outside its window.")

    completed.add(author_id)


resume_command = (
    "python data_science/collect_author_histories.py "
    f"--resume {run_id}"
)

print(f"Author-history run: {run_id}")
print(f"Already completed: {len(completed)}/{len(jobs)}")
print(f"Resume command: {resume_command}")

pending = [
    job for job in jobs if job["author_id"] not in completed
]
additional_completed = 0
active_author = None

try:
    api_key = ""

    if pending:
        api_key = getpass.getpass(
            "Paste your OpenAlex API key: "
        ).strip()

        if not api_key:
            raise ValueError("No API key was entered.")

    manifest["status"] = "running"
    save_json(manifest_file, manifest)

    for job in pending:
        if (
            args.limit is not None
            and additional_completed >= args.limit
        ):
            break

        active_author = job["author_id"]
        short_id = active_author.rsplit("/", 1)[-1]
        cutoff = date.fromisoformat(job["cutoff_exclusive"])
        last_date = cutoff - timedelta(days=1)

        print(
            f"\n[{len(completed) + 1}/{len(jobs)}] "
            f"{short_id}: {job['example_name']}"
        )
        print(f"Collecting articles dated before {cutoff}.")

        attempt_folder = raw_folder / short_id / timestamp()
        attempt_folder.mkdir(parents=True, exist_ok=False)

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
                    f"to_publication_date:{last_date},"
                    f"type:{queue['work_type']}"
                ),
                "corpus": queue["corpus"],
                "per_page": 100,
                "cursor": cursor,
                "select": "id,display_name,publication_date,type",
            }

            request_file = (
                attempt_folder
                / f"request_{page_number:04d}.json"
            )
            save_json(request_file, {
                "parameters": parameters,
                "requested_at_utc": utc_now(),
            })

            request = Request(
                "https://api.openalex.org/works?"
                + urlencode(parameters),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "quantum_phys_digest/0.5",
                },
            )

            time.sleep(1)

            with urlopen(request, timeout=60) as response:
                raw_bytes = response.read()

            (
                attempt_folder / f"page_{page_number:04d}.json"
            ).write_bytes(raw_bytes)

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

                if publication_date >= cutoff:
                    raise ValueError(
                        "An article is outside the window."
                    )

                if work["type"] != queue["work_type"]:
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
                f"{len(works)} articles collected"
            )

            next_cursor = data["meta"].get("next_cursor")

            if not results or next_cursor is None:
                break

            cursor = next_cursor

        if len(works) != expected_count:
            raise ValueError(
                "Collected articles differ from the API count."
            )

        result = {
            "author_id": active_author,
            "queue_sha256": queue_hash,
            "cutoff_exclusive": cutoff.isoformat(),
            "api_reported_count": expected_count,
            "validation_status": "passed",
            "completed_at_utc": utc_now(),
            "raw_attempt_folder": attempt_folder.relative_to(
                project_folder
            ).as_posix(),
            "works": sorted(
                works.values(),
                key=lambda work: (
                    work["publication_date"],
                    work["openalex_id"],
                ),
            ),
        }

        save_json(
            output_folder / f"{short_id}_history.json",
            result,
        )

        completed.add(active_author)
        additional_completed += 1

        manifest["authors_completed"] = len(completed)
        manifest["updated_at_utc"] = utc_now()
        save_json(manifest_file, manifest)

        print(
            f"  Saved; completed "
            f"{len(completed)}/{len(jobs)} authors"
        )

except (
    HTTPError,
    URLError,
    ValueError,
    KeyError,
    TypeError,
    OSError,
    KeyboardInterrupt,
) as error:
    manifest["status"] = (
        "paused" if isinstance(error, KeyboardInterrupt)
        else "failed"
    )
    manifest["active_author"] = active_author
    manifest["error_type"] = type(error).__name__
    manifest["authors_completed"] = len(completed)
    manifest["updated_at_utc"] = utc_now()
    save_json(manifest_file, manifest)

    if isinstance(error, HTTPError):
        message = f"OpenAlex returned HTTP {error.code}."
    elif isinstance(error, URLError):
        message = "A network connection failed."
    elif isinstance(error, KeyboardInterrupt):
        message = "Collection interrupted."
    else:
        message = str(error)

    raise SystemExit(
        f"\n{message}\nCompleted histories remain saved.\n"
        f"Resume with:\n{resume_command}"
    ) from None


manifest["status"] = (
    "completed" if len(completed) == len(jobs) else "paused"
)
manifest["authors_completed"] = len(completed)
manifest["updated_at_utc"] = utc_now()
save_json(manifest_file, manifest)

print(f"\nStatus: {manifest['status']}")
print(f"Completed: {len(completed)}/{len(jobs)}")
print(f"Results: {output_folder}")

if manifest["status"] != "completed":
    print(f"Resume with:\n{resume_command}")