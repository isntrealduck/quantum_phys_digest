import argparse
import getpass
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TOPIC_IDS = [
    "T10020",
    "T10622",
    "T10682",
    "T11414",
    "T11804",
]

YEARS = list(range(2015, 2020))
PAPERS_PER_YEAR = 1000
PAGE_SIZE = 100
SEED = 42

CONFIG = {
    "topic_ids": TOPIC_IDS,
    "topic_filter": "primary_topic.id",
    "work_type": "article",
    "years": YEARS,
    "papers_per_year": PAPERS_PER_YEAR,
    "page_size": PAGE_SIZE,
    "seed": SEED,
    "corpus": "core",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path, value):
    """Replace a file only after its new contents have been written."""
    temporary_file = path.with_name(path.name + ".tmp")
    temporary_file.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_file.replace(path)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume",
    metavar="RUN_ID",
    help="Resume an existing collection folder.",
)
args = parser.parse_args()

project_folder = Path(__file__).resolve().parent.parent
collection_root = (
    project_folder / "data" / "raw" / "openalex_historical"
)

if args.resume:
    run_id = args.resume

    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise SystemExit("Supply a folder name, not a full path.")

    output_folder = collection_root / run_id
    checkpoint = read_json(output_folder / "checkpoint.json")

    manifest = checkpoint["manifest"]
    papers = checkpoint["papers"]

    if any(manifest.get(key) != value for key, value in CONFIG.items()):
        raise SystemExit("This collection uses different settings.")

    if manifest["status"] == "completed":
        raise SystemExit("This collection is already completed.")

    print(f"Resuming {run_id}: {len(papers)} records already saved.")

else:
    run_id = timestamp()
    output_folder = collection_root / run_id
    output_folder.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "status": "running",
        "endpoint": "https://api.openalex.org/works",
        **CONFIG,
        "requests": [],
    }

    papers = []


def save_progress():
    manifest["total_records"] = len(papers)
    manifest["unique_ids"] = len({paper["id"] for paper in papers})

    # This single file is the authoritative resume checkpoint.
    write_json_atomic(
        output_folder / "checkpoint.json",
        {"manifest": manifest, "papers": papers},
    )

    # Separate files remain convenient for later processing.
    write_json_atomic(output_folder / "works.json", papers)
    write_json_atomic(output_folder / "manifest.json", manifest)


seen_ids = {paper["id"] for paper in papers}

if len(seen_ids) != len(papers):
    raise SystemExit("The checkpoint contains duplicate paper IDs.")

completed_pages = {
    (entry["year"], entry["page"])
    for entry in manifest["requests"]
}

if len(completed_pages) != len(manifest["requests"]):
    raise SystemExit("The checkpoint contains duplicate page entries.")

if len(papers) != sum(
    entry["records_received"] for entry in manifest["requests"]
):
    raise SystemExit("Checkpoint record counts are inconsistent.")


api_key = getpass.getpass("Paste your OpenAlex API key: ").strip()

if not api_key:
    raise SystemExit("No API key was entered.")

manifest["status"] = "running"
manifest["last_started_at_utc"] = utc_now()
manifest.pop("error_type", None)
manifest.pop("http_status", None)
save_progress()

print(f"Collection folder: {output_folder}")

pages_per_year = PAPERS_PER_YEAR // PAGE_SIZE

try:
    for year in YEARS:
        for page in range(1, pages_per_year + 1):
            if (year, page) in completed_pages:
                continue

            parameters = {
                "filter": (
                    f"primary_topic.id:{'|'.join(TOPIC_IDS)},"
                    f"publication_year:{year},"
                    "type:article"
                ),
                "corpus": "core",
                "sample": PAPERS_PER_YEAR,
                "seed": SEED,
                "per_page": PAGE_SIZE,
                "page": page,
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

            print(
                f"Collecting {year}, page {page}/{pages_per_year}..."
            )

            time.sleep(1)

            with urlopen(request, timeout=60) as response:
                raw_response = response.read()

            retrieved_at = utc_now()

            # Unique filenames preserve unsuccessful attempts too.
            response_file = output_folder / (
                f"response_{year}_page_{page:02d}_{timestamp()}.json"
            )
            response_file.write_bytes(raw_response)

            data = json.loads(raw_response)
            batch = data["results"]

            if len(batch) != PAGE_SIZE:
                raise ValueError(
                    f"Expected {PAGE_SIZE} papers for {year}, "
                    f"page {page}; received {len(batch)}."
                )

            batch_ids = set()

            for paper in batch:
                paper_id = paper["id"]
                topic_id = (
                    (paper.get("primary_topic") or {}).get("id") or ""
                ).rsplit("/", 1)[-1]

                if paper_id in seen_ids or paper_id in batch_ids:
                    raise ValueError(
                        f"Duplicate OpenAlex ID: {paper_id}"
                    )

                if paper.get("publication_year") != year:
                    raise ValueError(
                        f"Unexpected year for {paper_id}"
                    )

                if paper.get("type") != "article":
                    raise ValueError(
                        f"Unexpected type for {paper_id}"
                    )

                if topic_id not in TOPIC_IDS:
                    raise ValueError(
                        f"Unexpected topic for {paper_id}"
                    )

                batch_ids.add(paper_id)

            papers.extend(batch)
            seen_ids.update(batch_ids)

            manifest["requests"].append({
                "year": year,
                "page": page,
                "parameters": parameters,
                "retrieved_at_utc": retrieved_at,
                "response_file": response_file.name,
                "records_received": len(batch),
            })

            completed_pages.add((year, page))
            save_progress()

            print(f"  Saved {len(papers)}/5000 records.")

    expected_total = len(YEARS) * PAPERS_PER_YEAR

    if len(papers) != expected_total:
        raise ValueError("The final collection size is incorrect.")

    for year in YEARS:
        year_count = sum(
            paper["publication_year"] == year for paper in papers
        )

        if year_count != PAPERS_PER_YEAR:
            raise ValueError(f"Incorrect final count for {year}.")

except (
    HTTPError,
    URLError,
    ValueError,
    KeyError,
    TypeError,
    OSError,
    KeyboardInterrupt,
) as error:
    manifest["status"] = "interrupted" if isinstance(
        error, KeyboardInterrupt
    ) else "failed"

    manifest["stopped_at_utc"] = utc_now()
    manifest["error_type"] = type(error).__name__

    if isinstance(error, HTTPError):
        manifest["http_status"] = error.code
        message = (
            f"OpenAlex returned HTTP {error.code}. "
            "401/403: check your key. "
            "429: request limit or daily budget reached."
        )
    elif isinstance(error, URLError):
        message = "Could not connect to OpenAlex."
    elif isinstance(error, KeyboardInterrupt):
        message = "Collection interrupted."
    else:
        message = str(error)

    save_progress()

    raise SystemExit(
        f"{message}\n"
        f"Saved records: {len(papers)}\n"
        f"Resume with:\n"
        f"python fetch_openalex_large.py --resume {run_id}"
    ) from None


manifest["status"] = "completed"
manifest["completed_at_utc"] = utc_now()
save_progress()

print(f"\nCollection completed: {len(papers)} records.")
print(f"Unique OpenAlex IDs: {len(seen_ids)}")
print(f"Collection folder: {output_folder}")