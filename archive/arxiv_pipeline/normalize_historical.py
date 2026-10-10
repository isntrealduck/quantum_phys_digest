import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def clean_text(value):
    """Normalize Unicode and replace repeated whitespace."""
    value = unicodedata.normalize("NFC", value or "")
    return " ".join(value.split())


def normalize_date(value):
    """Convert a timestamp into a consistent UTC representation."""
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value}")
    return date.astimezone(timezone.utc).isoformat()


# Require an explicit collection folder to avoid mixing different runs.
if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python normalize_historical.py COLLECTION_FOLDER_NAME"
    )

run_id = sys.argv[1]
if Path(run_id).name != run_id or run_id in {".", ".."}:
    raise ValueError("Supply a folder name, not a full path.")

project_folder = Path(__file__).resolve().parent
source_folder = project_folder / "data" / "raw" / "historical" / run_id

manifest = json.loads(
    (source_folder / "manifest.json").read_text(encoding="utf-8")
)

if manifest["status"] != "completed":
    raise ValueError("This collection run is incomplete.")

papers = []
seen_ids = set()

for request_record in manifest["requests"]:
    if not request_record["validation_passed"]:
        raise ValueError("A source response failed collection validation.")

    source_file = source_folder / request_record["raw_file"]
    root = ET.fromstring(source_file.read_bytes())
    entries = root.findall("atom:entry", NS)

    if len(entries) != request_record["returned_count"]:
        raise ValueError(f"Unexpected record count in {source_file.name}")

    for entry in entries:
        source_url = entry.findtext("atom:id", "", NS)
        versioned_id = urlparse(source_url).path.removeprefix("/abs/")

        # Separate the permanent paper ID from its version number.
        match = re.fullmatch(r"(\d{4}\.\d{4,5})v(\d+)", versioned_id)
        if not match:
            raise ValueError(f"Unexpected arXiv identifier: {source_url}")

        arxiv_id = match.group(1)
        version = int(match.group(2))

        if arxiv_id in seen_ids:
            raise ValueError(f"Duplicate paper ID: {arxiv_id}")
        seen_ids.add(arxiv_id)

        title = clean_text(entry.findtext("atom:title", "", NS))
        abstract = clean_text(entry.findtext("atom:summary", "", NS))

        if not title or not abstract:
            raise ValueError(f"Missing title or abstract: {arxiv_id}")

        submitted_at = normalize_date(
            entry.findtext("atom:published", "", NS)
        )
        updated_at = normalize_date(
            entry.findtext("atom:updated", "", NS)
        )

        if datetime.fromisoformat(submitted_at).year != request_record["year"]:
            raise ValueError(f"Unexpected submission year: {arxiv_id}")

        if datetime.fromisoformat(updated_at) < datetime.fromisoformat(submitted_at):
            raise ValueError(f"Update precedes initial submission: {arxiv_id}")

        authors = []
        for author in entry.findall("atom:author", NS):
            name = clean_text(author.findtext("atom:name", "", NS))
            if not name:
                raise ValueError(f"Missing author name: {arxiv_id}")

            authors.append({
                "name": name,
                "affiliations": [
                    clean_text(affiliation.text)
                    for affiliation in author.findall("arxiv:affiliation", NS)
                    if clean_text(affiliation.text)
                ],
            })

        if not authors:
            raise ValueError(f"No authors found: {arxiv_id}")

        categories = sorted({
            category.get("term")
            for category in entry.findall("atom:category", NS)
            if category.get("term")
        })

        primary_element = entry.find("arxiv:primary_category", NS)
        primary_category = (
            primary_element.get("term") if primary_element is not None else None
        )

        if "quant-ph" not in categories or primary_category not in categories:
            raise ValueError(f"Invalid categories: {arxiv_id}")

        doi = clean_text(entry.findtext("arxiv:doi", "", NS))
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
        doi = doi or None

        papers.append({
            "arxiv_id": arxiv_id,
            "version": version,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "version_url": f"https://arxiv.org/abs/{arxiv_id}v{version}",
            "title": title,
            "abstract": abstract,
            "submitted_at": submitted_at,
            "updated_at": updated_at,
            "authors": authors,
            "categories": categories,
            "primary_category": primary_category,
            "doi": doi,
            "source_file": request_record["raw_file"],
            "source_run": run_id,
            "retrieved_at_utc": request_record["retrieved_at_utc"],
            "publication_time_features_verified": False,
        })

if len(papers) != manifest["total_records"]:
    raise ValueError("Normalized count differs from the collection manifest.")

papers.sort(key=lambda paper: (paper["submitted_at"], paper["arxiv_id"]))

report = {
    "source_run": run_id,
    "paper_count": len(papers),
    "unique_paper_ids": len(seen_ids),
    "papers_without_doi": sum(p["doi"] is None for p in papers),
    "papers_without_any_affiliation": sum(
        not any(author["affiliations"] for author in p["authors"])
        for p in papers
    ),
    "papers_with_returned_version_above_1": sum(
        p["version"] > 1 for p in papers
    ),
    "validation_passed": True,
}

output_folder = project_folder / "data" / "processed" / run_id
output_folder.mkdir(parents=True, exist_ok=True)

for filename, content in [
    ("papers.json", papers),
    ("validation_report.json", report),
]:
    (output_folder / filename).write_text(
        json.dumps(content, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

print(json.dumps(report, indent=2))
print(f"\nSaved normalized records to: {output_folder}")