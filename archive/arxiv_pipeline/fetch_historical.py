import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

# Collection settings.
YEARS = range(2015, 2020)
PAPERS_PER_YEAR = 100
API_URL = "https://export.arxiv.org/api/query"

# XML namespaces used by arXiv.
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# Give each collection run its own folder.
project_folder = Path(__file__).resolve().parent
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output_folder = project_folder / "data" / "raw" / "historical" / run_id
output_folder.mkdir(parents=True, exist_ok=False)

manifest_file = output_folder / "manifest.json"

manifest = {
    "run_id": run_id,
    "category": "quant-ph",
    "sampling_method": "earliest 100 submissions per year",
    "metadata_version": "version returned by search; not guaranteed v1",
    "status": "in_progress",
    "requests": [],
}


def save_manifest():
    """Save the current collection record."""
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


save_manifest()

for year in YEARS:
    # Pause before each request.
    time.sleep(4)

    # Select submissions within this calendar year.
    query = (
        "cat:quant-ph AND "
        f"submittedDate:[{year}01010000 TO {year}12312359]"
    )

    parameters = {
        "search_query": query,
        "start": 0,
        "max_results": PAPERS_PER_YEAR,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }

    url = f"{API_URL}?{urlencode(parameters)}"
    request = Request(
        url,
        headers={"User-Agent": "quantum_phys_digest/0.1"},
    )

    print(f"Requesting {PAPERS_PER_YEAR} papers from {year}...")

    with urlopen(request, timeout=60) as response:
        raw_xml = response.read()

    # Preserve the response before parsing or cleaning it.
    filename = f"quant_ph_{year}.xml"
    (output_folder / filename).write_bytes(raw_xml)

    request_record = {
        "year": year,
        "parameters": parameters,
        "url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": filename,
        "validation_passed": False,
    }

    manifest["requests"].append(request_record)
    save_manifest()

    # Read the XML only to check the collection.
    root = ET.fromstring(raw_xml)
    entries = root.findall("atom:entry", NS)

    for entry in entries:
        entry_url = entry.findtext("atom:id", "", NS)

        if "/api/errors" in entry_url:
            message = entry.findtext("atom:summary", "", NS)
            raise RuntimeError(f"arXiv API error: {message}")

    request_record["returned_count"] = len(entries)
    request_record["total_matching_results"] = int(
        root.findtext("opensearch:totalResults", "0", NS)
    )
    save_manifest()

    if len(entries) != PAPERS_PER_YEAR:
        raise RuntimeError(
            f"{year}: expected {PAPERS_PER_YEAR} papers, "
            f"but received {len(entries)}. Raw response saved."
        )

    for entry in entries:
        submitted_at = entry.findtext("atom:published", "", NS)
        submitted_year = datetime.fromisoformat(
            submitted_at.replace("Z", "+00:00")
        ).year

        if submitted_year != year:
            raise ValueError(
                f"Unexpected submission date: {submitted_at}"
            )

        categories = [
            category.get("term")
            for category in entry.findall("atom:category", NS)
        ]

        if "quant-ph" not in categories:
            raise ValueError("A returned paper lacks quant-ph category.")

    request_record["validation_passed"] = True
    save_manifest()

    print(f"{year}: saved and checked {len(entries)} papers.")

manifest["status"] = "completed"
manifest["total_records"] = sum(
    record["returned_count"] for record in manifest["requests"]
)
save_manifest()

print(f"\nCollection completed: {manifest['total_records']} records.")
print(f"Saved to: {output_folder}")