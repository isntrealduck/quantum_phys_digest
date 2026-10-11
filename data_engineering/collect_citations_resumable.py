import argparse
import dotenv
import getpass
import hashlib
import json
import os
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


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    """Replace the destination only after writing the complete new file."""
    temporary_file = path.with_name(path.name + ".tmp")
    temporary_file.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_file.replace(path)


def fifth_anniversary(start):
    try:
        return start.replace(year=start.year + 5)
    except ValueError:
        return start.replace(year=start.year + 5, day=28)


def check_folder_name(value):
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError("Supply a folder name, not a full path.")
    return value


parser = argparse.ArgumentParser()
parser.add_argument("collection_run", type=check_folder_name)
parser.add_argument("--resume", type=check_folder_name, metavar="CITATION_RUN")
parser.add_argument(
    "--limit",
    type=int,
    help="Stop after this many new labels, for testing resumability.",
)
args = parser.parse_args()

if args.limit is not None and args.limit < 1:
    raise SystemExit("--limit must be a positive integer.")

collection_run = args.collection_run
project_folder = Path(__file__).resolve().parent.parent

collection_folder = (
    project_folder
    / "data"
    / "processed"
    / "openalex_historical"
    / collection_run
)

input_bytes = (collection_folder / "papers.json").read_bytes()
input_hash = hashlib.sha256(input_bytes).hexdigest()

papers = sorted(
    json.loads(input_bytes),
    key=lambda paper: (
        paper["publication_date"],
        paper["openalex_id"],
    ),
)

ids = [paper["openalex_id"] for paper in papers]

if not papers or len(ids) != len(set(ids)):
    raise SystemExit("The collection is empty or contains duplicate IDs.")

papers_by_id = {paper["openalex_id"]: paper for paper in papers}
citation_run = args.resume or timestamp()

raw_folder = (
    project_folder
    / "data"
    / "raw"
    / "openalex_five_year_citations"
    / collection_run
    / citation_run
)

output_folder = collection_folder / "citation_labels" / citation_run
labels_file = output_folder / "labels.json"
manifest_file = raw_folder / "manifest.json"

if args.resume:
    manifest = read_json(manifest_file)

    if (
        manifest.get("collection_run") != collection_run
        or manifest.get("input_sha256") != input_hash
        or manifest.get("corpus") != "core"
        or manifest.get("collector_version") != 1
    ):
        raise SystemExit("The input or collector settings differ from this run.")

    labels = read_json(labels_file)

else:
    raw_folder.mkdir(parents=True, exist_ok=False)
    output_folder.mkdir(parents=True, exist_ok=False)

    manifest = {
        "collector_version": 1,
        "collection_run": collection_run,
        "citation_run": citation_run,
        "input_sha256": input_hash,
        "started_at_utc": utc_now(),
        "status": "running",
        "corpus": "core",
        "window_definition": (
            "publication_date <= citing_publication_date < fifth_anniversary"
        ),
        "february_29_rule": "Use February 28 for the fifth anniversary.",
        "selected_openalex_ids": ids,
        "requests": [],
    }

    labels = []
    save_json(labels_file, labels)
    save_json(manifest_file, manifest)


# Check saved labels before trusting them during a resume.
completed_ids = set()

for label in labels:
    target_id = label["openalex_id"]

    if target_id not in papers_by_id or target_id in completed_ids:
        raise SystemExit("Saved labels contain unexpected or duplicate IDs.")

    paper = papers_by_id[target_id]
    start = date.fromisoformat(paper["publication_date"])
    end = fifth_anniversary(start)
    count = label["five_year_citation_count"]
    filename = label["citing_works_file"]

    if Path(filename).name != filename:
        raise SystemExit("Invalid saved citing-works filename.")

    works = read_json(output_folder / filename)
    citing_ids = [work["openalex_id"] for work in works]

    if (
        type(count) is not int
        or count < 0
        or count != len(works)
        or count + label.get("excluded_self_reference_count", 0)
        != label["api_reported_window_count"]
        or len(citing_ids) != len(set(citing_ids))
        or target_id in citing_ids
        or label["validation_status"] != "passed"
        or label["publication_date"] != start.isoformat()
        or label["window_end_exclusive"] != end.isoformat()
        or any(
            not start <= date.fromisoformat(work["publication_date"]) < end
            for work in works
        )
    ):
        raise SystemExit(f"Invalid saved evidence for {target_id}.")

    completed_ids.add(target_id)

if len(labels) == len(papers):
    manifest["status"] = "completed"
    manifest["labels_created"] = len(labels)
    manifest.setdefault("completed_at_utc", utc_now())
    save_json(manifest_file, manifest)
    raise SystemExit("All labels are already completed.")

dotenv.load_dotenv(project_folder / ".env")
api_key = os.getenv("OPENALEX_API_KEY", "").strip() or getpass.getpass(
    "Paste your OpenAlex API key: "
).strip()

if not api_key:
    raise SystemExit("No API key was entered.")

manifest["status"] = "running"
manifest["last_started_at_utc"] = utc_now()
manifest.pop("error_type", None)
manifest.pop("http_status", None)
save_json(manifest_file, manifest)

resume_command = (
    f"python collect_citations_resumable.py {collection_run} "
    f"--resume {citation_run}"
)

print(f"Citation run: {citation_run}")
print(f"Already completed: {len(labels)}/{len(papers)}")
print(f"Resume command: {resume_command}")

new_labels = 0

try:
    for number, paper in enumerate(papers, start=1):
        target_id = paper["openalex_id"]

        if target_id in completed_ids:
            continue

        short_id = target_id.rsplit("/", 1)[-1]

        if not re.fullmatch(r"W\d+", short_id):
            raise ValueError("Invalid OpenAlex work ID.")

        start = date.fromisoformat(paper["publication_date"])
        end = fifth_anniversary(start)
        last_date = end - timedelta(days=1)

        if end > datetime.now(timezone.utc).date():
            raise ValueError("The five-year window has not finished.")

        print(f"\n[{number}/{len(papers)}] {short_id}: {paper['title']}")

        # Each attempt gets its own folder, preserving partial earlier attempts.
        attempt_folder = raw_folder / short_id / timestamp()
        attempt_folder.mkdir(parents=True, exist_ok=False)

        cursor = "*"
        used_cursors = set()
        citing_works = {}
        expected_count = None
        page_number = 0

        while True:
            if cursor in used_cursors:
                raise ValueError("The API repeated a cursor.")

            used_cursors.add(cursor)
            page_number += 1

            parameters = {
                "filter": (
                    f"cites:{short_id},"
                    f"from_publication_date:{start},"
                    f"to_publication_date:{last_date}"
                ),
                "corpus": "core",
                "per_page": 100,
                "cursor": cursor,
                "select": "id,display_name,publication_date,referenced_works",
            }

            request = Request(
                "https://api.openalex.org/works?" + urlencode(parameters),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "quantum_phys_digest/0.4",
                },
            )

            time.sleep(1)

            with urlopen(request, timeout=60) as response:
                raw_bytes = response.read()

            raw_file = attempt_folder / f"page_{page_number:04d}.json"
            raw_file.write_bytes(raw_bytes)

            manifest["requests"].append({
                "target_openalex_id": target_id,
                "parameters": parameters,
                "retrieved_at_utc": utc_now(),
                "raw_response_file": raw_file.relative_to(
                    project_folder
                ).as_posix(),
            })
            save_json(manifest_file, manifest)

            data = json.loads(raw_bytes)
            page_count = data["meta"]["count"]

            if expected_count is None:
                expected_count = page_count
            elif expected_count != page_count:
                raise ValueError("The API count changed during pagination.")

            results = data["results"]

            for work in results:
                citing_id = work["id"]
                citing_date = date.fromisoformat(work["publication_date"])

                if not start <= citing_date < end:
                    raise ValueError("A citing work is outside the window.")

                if target_id not in (work.get("referenced_works") or []):
                    raise ValueError("A returned work does not cite the target.")

                if citing_id in citing_works:
                    raise ValueError("Duplicate citing ID.")

                citing_works[citing_id] = {
                    "openalex_id": citing_id,
                    "title": work.get("display_name"),
                    "publication_date": work["publication_date"],
                }

            print(
                f"  Page {page_number}: "
                f"{len(citing_works)} unique citing works"
            )

            next_cursor = data["meta"].get("next_cursor")

            if not results or next_cursor is None:
                break

            cursor = next_cursor

        if len(citing_works) != expected_count:
            raise ValueError(
                f"Collected {len(citing_works)} unique IDs, "
                f"but the API reported {expected_count}."
            )

        excluded_self_reference_count = int(
            citing_works.pop(target_id, None) is not None
        )

        if excluded_self_reference_count:
            print("  Excluded the target paper's own ID from the citation count.")

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

        label = {
            "openalex_id": target_id,
            "title": paper["title"],
            "publication_date": start.isoformat(),
            "window_end_exclusive": end.isoformat(),
            "five_year_citation_count": len(citing_works),
            "historical_current_citation_count": paper["current_citation_count"],
            "api_reported_window_count": expected_count,
            "excluded_self_reference_count": excluded_self_reference_count,
            "validation_status": "passed",
            "manual_review": "pending",
            "citing_works_file": citations_file.name,
            "raw_attempt_folder": attempt_folder.relative_to(
                project_folder
            ).as_posix(),
            "completed_at_utc": utc_now(),
        }

        # Save before marking this paper completed in memory.
        updated_labels = labels + [label]
        save_json(labels_file, updated_labels)
        labels = updated_labels
        completed_ids.add(target_id)

        manifest["labels_created"] = len(labels)
        save_json(manifest_file, manifest)
        new_labels += 1

        print(
            f"  Target: {len(citing_works)}; "
            f"completed {len(labels)}/{len(papers)}"
        )

        if args.limit is not None and new_labels >= args.limit:
            break

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
        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    )
    manifest["stopped_at_utc"] = utc_now()
    manifest["error_type"] = type(error).__name__

    if isinstance(error, HTTPError):
        manifest["http_status"] = error.code
        message = f"OpenAlex returned HTTP {error.code}."
    elif isinstance(error, URLError):
        message = "A network connection failed."
    elif isinstance(error, KeyboardInterrupt):
        message = "Collection interrupted."
    else:
        message = str(error)

    save_json(manifest_file, manifest)

    raise SystemExit(
        f"{message}\n"
        f"Completed labels remain saved.\nResume with:\n{resume_command}"
    ) from None


manifest["labels_created"] = len(labels)

if len(labels) == len(papers):
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = utc_now()
else:
    manifest["status"] = "paused"
    manifest["paused_at_utc"] = utc_now()

save_json(manifest_file, manifest)

print(f"\nStatus: {manifest['status']}")
print(f"Labels created: {len(labels)}/{len(papers)}")
print(f"Results: {labels_file}")

if manifest["status"] == "paused":
    print(f"Resume with:\n{resume_command}")