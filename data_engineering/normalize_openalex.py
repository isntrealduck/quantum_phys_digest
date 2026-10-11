import json
import re
import sys
import unicodedata
from pathlib import Path


def clean_text(value):
    """Normalize Unicode and whitespace, preserving capitalization."""
    if value is None:
        return None

    text = unicodedata.normalize("NFC", value)
    return " ".join(text.split()) or None


def normalize_doi(value):
    """Convert a DOI URL to a lowercase DOI identifier."""
    if not value:
        return None

    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    ).lower()


def reconstruct_abstract(index):
    """Put abstract words back into their recorded positions."""
    if not index:
        return None

    words_by_position = {}

    for word, positions in index.items():
        for position in positions:
            if type(position) is not int or position < 0:
                raise ValueError("Invalid abstract position.")

            if position in words_by_position:
                raise ValueError("Repeated abstract position.")

            words_by_position[position] = word

    positions = sorted(words_by_position)

    if positions != list(range(len(positions))):
        raise ValueError("Abstract positions contain gaps.")

    return clean_text(
        " ".join(words_by_position[position] for position in positions)
    )


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python normalize_openalex.py COLLECTION_FOLDER"
    )

run_id = sys.argv[1]

if Path(run_id).name != run_id or run_id in {".", ".."}:
    raise ValueError("Supply a collection folder name, not a full path.")

project_folder = Path(__file__).resolve().parent.parent

input_folder = (
    project_folder / "data" / "raw" / "openalex_historical" / run_id
)

manifest = json.loads(
    (input_folder / "manifest.json").read_text(encoding="utf-8")
)

if manifest["status"] != "completed":
    raise SystemExit("The collection did not finish successfully.")

raw_papers = json.loads(
    (input_folder / "works.json").read_text(encoding="utf-8")
)

# Find each paper in the response pages recorded in the manifest.
requests_by_id = {}
source_works_by_id = {}

for request in manifest["requests"]:
    response_filename = request["response_file"]

    if Path(response_filename).name != response_filename:
        raise ValueError("Expected a response filename, not a path.")

    response_file = input_folder / response_filename
    response = json.loads(response_file.read_text(encoding="utf-8"))
    batch = response["results"]

    if len(batch) != request["records_received"]:
        raise ValueError(
            f"Response count differs from manifest: {response_filename}"
        )

    for source_work in batch:
        work_id = source_work["id"]

        if work_id in requests_by_id:
            raise ValueError(
                f"Paper appears in multiple response pages: {work_id}"
            )

        if source_work["publication_year"] != request["year"]:
            raise ValueError(
                f"Paper year differs from its request: {work_id}"
            )

        requests_by_id[work_id] = request
        source_works_by_id[work_id] = source_work

raw_ids = [work["id"] for work in raw_papers]

if len(raw_ids) != len(set(raw_ids)):
    raise ValueError("Duplicate IDs in works.json.")

if set(raw_ids) != set(requests_by_id):
    raise ValueError("Response-page IDs differ from works.json IDs.")

for work in raw_papers:
    if work != source_works_by_id[work["id"]]:
        raise ValueError(
            f"Record differs from its original response: {work['id']}"
        )

papers = []
invalid_abstract_ids = []

for work in raw_papers:
    try:
        abstract = reconstruct_abstract(
            work.get("abstract_inverted_index")
        )
    except ValueError:
        abstract = None
        invalid_abstract_ids.append(work["id"])

    authors = []

    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}

        institutions = [
            {
                "id": institution.get("id"),
                "name": clean_text(institution.get("display_name")),
                "country_code": institution.get("country_code"),
            }
            for institution in authorship.get("institutions") or []
        ]

        authors.append({
            "id": author.get("id"),
            "name": clean_text(author.get("display_name")),
            "orcid": author.get("orcid"),
            "author_position": authorship.get("author_position"),
            "institutions": institutions,
        })

    topic = work.get("primary_topic") or {}
    source_request = requests_by_id[work["id"]]

    paper = {
        "openalex_id": work["id"],
        "doi": normalize_doi(work.get("doi")),
        "title": clean_text(work.get("title")),
        "abstract": abstract,
        "publication_date": work["publication_date"],
        "publication_year": work["publication_year"],
        "type": work["type"],
        "language": work.get("language"),
        "authors": authors,
        "observed_primary_topic": {
            "id": topic.get("id"),
            "name": clean_text(topic.get("display_name")),
        },
        "current_citation_count": work.get("cited_by_count"),
        "five_year_citation_count": None,
        "publication_time_features_verified": False,
        "observed_is_retracted": work.get("is_retracted"),
        "observed_is_authors_truncated": work.get("is_authors_truncated"),
        "quality_flags": {
            "missing_title": clean_text(work.get("title")) is None,
            "abstract_unavailable": abstract is None,
            "abstract_index_invalid": work["id"] in invalid_abstract_ids,
            "missing_authorships": not authors,
            "without_recorded_institutions": not any(
                author["institutions"] for author in authors
            ),
           "missing_doi": normalize_doi(work.get("doi")) is None,
        },
        "provenance": {
            "collection_run": run_id,
            "raw_response_file": source_request["response_file"],
            "retrieved_at_utc": source_request["retrieved_at_utc"],
        },
    }

    papers.append(paper)


# Check that normalization preserved all records.
unique_ids = {paper["openalex_id"] for paper in papers}

if len(papers) != manifest["total_records"]:
    raise ValueError("Record count differs from the collection manifest.")

if len(unique_ids) != len(papers):
    raise ValueError("Duplicate OpenAlex IDs found.")

report = {
    "collection_run": run_id,
    "records_normalized": len(papers),
    "unique_openalex_ids": len(unique_ids),
    "abstracts_reconstructed": sum(
        paper["abstract"] is not None for paper in papers
    ),
    "abstracts_unavailable": sum(
        paper["abstract"] is None for paper in papers
    ),
    "invalid_abstract_ids": invalid_abstract_ids,
    "five_year_labels_created": 0,
}

output_folder = (
    project_folder
    / "data"
    / "processed"
    / "openalex_historical"
    / run_id
)
output_folder.mkdir(parents=True, exist_ok=True)

(output_folder / "papers.json").write_text(
    json.dumps(papers, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

(output_folder / "normalization_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))
print(f"\nNormalized papers: {output_folder / 'papers.json'}")