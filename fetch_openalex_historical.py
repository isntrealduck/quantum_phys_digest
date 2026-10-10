import getpass
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# Define the initial subject scope.
TOPIC_IDS = [
    "T10020",
    "T10622",
    "T10682",
    "T11414",
    "T11804",
]

YEARS = range(2015, 2020)
PAPERS_PER_YEAR = 100
SEED = 42


# Read the key without displaying or saving it.
api_key = getpass.getpass(
    "Paste your OpenAlex API key and press Enter: "
).strip()

if not api_key:
    raise SystemExit("No API key was entered.")


# Create a separate folder for this collection.
project_folder = Path(__file__).resolve().parent
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

output_folder = (
    project_folder / "data" / "raw" / "openalex_historical" / run_id
)
output_folder.mkdir(parents=True, exist_ok=False)

manifest = {
    "run_id": run_id,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "running",
    "endpoint": "https://api.openalex.org/works",
    "topic_ids": TOPIC_IDS,
    "topic_filter": "primary_topic.id",
    "work_type": "article",
    "years": list(YEARS),
    "papers_per_year": PAPERS_PER_YEAR,
    "seed": SEED,
    "requests": [],
}

papers = []
seen_ids = set()


def save_progress():
    """Preserve collected records and collection details."""
    (output_folder / "works.json").write_text(
        json.dumps(papers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


save_progress()

try:
    for year in YEARS:
        # The pipe symbol means any one of these topics.
        topic_filter = "|".join(TOPIC_IDS)

        parameters = {
            "filter": (
                f"primary_topic.id:{topic_filter},"
                f"publication_year:{year},"
                "type:article"
            ),
            "sample": PAPERS_PER_YEAR,
            "seed": SEED,
            "per_page": PAPERS_PER_YEAR,
        }

        url = "https://api.openalex.org/works?" + urlencode(parameters)

        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "quantum_phys_digest/0.2",
            },
        )

        print(f"Collecting {year}...")

        with urlopen(request, timeout=60) as response:
            raw_response = response.read()

        # Save the complete response before extracting records.
        response_file = output_folder / f"response_{year}.json"
        response_file.write_bytes(raw_response)

        data = json.loads(raw_response)
        batch = data["results"]

        if len(batch) != PAPERS_PER_YEAR:
            raise ValueError(
                f"Expected {PAPERS_PER_YEAR} records for {year}, "
                f"received {len(batch)}."
            )

        batch_ids = set()

        for paper in batch:
            paper_id = paper["id"]
            topic_id = (
                (paper.get("primary_topic") or {}).get("id") or ""
            ).rsplit("/", 1)[-1]

            if paper_id in seen_ids or paper_id in batch_ids:
                raise ValueError(f"Duplicate OpenAlex ID: {paper_id}")

            if paper.get("publication_year") != year:
                raise ValueError(f"Unexpected year for {paper_id}")

            if paper.get("type") != "article":
                raise ValueError(f"Unexpected work type for {paper_id}")

            if topic_id not in TOPIC_IDS:
                raise ValueError(f"Unexpected primary topic for {paper_id}")

            batch_ids.add(paper_id)

        papers.extend(batch)
        seen_ids.update(batch_ids)

        manifest["requests"].append({
            "year": year,
            "parameters": parameters,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "response_file": response_file.name,
            "records_received": len(batch),
        })

        save_progress()
        print(f"  Saved {len(batch)} records.")

        time.sleep(1)

except (HTTPError, URLError, ValueError, KeyError) as error:
    manifest["status"] = "failed"
    manifest["error_type"] = type(error).__name__
    save_progress()

    if isinstance(error, HTTPError):
        message = (
            f"OpenAlex returned HTTP {error.code}. "
            "401/403: check your key. "
            "429: request limit or daily budget reached."
        )
    elif isinstance(error, URLError):
        message = "Could not connect to OpenAlex."
    else:
        message = str(error)

    raise SystemExit(
        f"{message}\nSaved progress: {output_folder}"
    ) from None


manifest["status"] = "completed"
manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest["total_records"] = len(papers)
manifest["unique_ids"] = len(seen_ids)
save_progress()

print(f"\nCollection completed: {len(papers)} records.")
print(f"Unique OpenAlex IDs: {len(seen_ids)}")
print(f"Collection folder: {output_folder}")