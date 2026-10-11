import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


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

dataset_bytes = (dataset_folder / "papers_labeled.json").read_bytes()
papers = json.loads(dataset_bytes)

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

# Ensure these splits belong to this exact dataset.
if hashlib.sha256(dataset_bytes).hexdigest() != splits["dataset_sha256"]:
    raise ValueError("The dataset changed after the splits were created.")

train_ids = set(splits["ids_by_split"]["train"])

train = [
    paper for paper in papers
    if paper["openalex_id"] in train_ids
]

if len(train) != len(train_ids):
    raise ValueError("Some training papers are missing.")

counts = [paper["five_year_citation_count"] for paper in train]

if any(type(count) is not int or count < 0 for count in counts):
    raise ValueError("Invalid citation targets.")

topics = Counter(
    (paper.get("observed_primary_topic") or {}).get("name") or "Unknown"
    for paper in train
)

missing_fields = {
    "title": sum(not paper.get("title") for paper in train),
    "abstract": sum(not paper.get("abstract") for paper in train),
    "authorships": sum(not paper.get("authors") for paper in train),
    "institutions": sum(
        not any(
            author.get("institutions")
            for author in paper.get("authors") or []
        )
        for paper in train
    ),
    "doi": sum(not paper.get("doi") for paper in train),
}

missing_report = {
    field: {
        "records": count,
        "percentage": round(100 * count / len(train), 2),
    }
    for field, count in missing_fields.items()
}

topic_report = {}

for topic in sorted(topics):
    topic_counts = [
        paper["five_year_citation_count"]
        for paper in train
        if (
            (paper.get("observed_primary_topic") or {}).get("name")
            or "Unknown"
        ) == topic
    ]

    topic_report[topic] = {
        "papers": len(topic_counts),
        "median_citations": statistics.median(topic_counts),
        "mean_citations": round(statistics.mean(topic_counts), 2),
        "zero_citation_percentage": round(
            100 * sum(count == 0 for count in topic_counts)
            / len(topic_counts),
            2,
        ),
    }

top_papers = sorted(
    train,
    key=lambda paper: (
        -paper["five_year_citation_count"],
        paper["openalex_id"],
    ),
)[:5]

report = {
    "split": "train",
    "records": len(train),
    "records_by_year": dict(sorted(Counter(
        paper["publication_year"] for paper in train
    ).items())),
    "five_year_citations": {
        "minimum": min(counts),
        "median": statistics.median(counts),
        "mean": round(statistics.mean(counts), 2),
        "maximum": max(counts),
        "zero_count": sum(count == 0 for count in counts),
        "zero_percentage": round(
            100 * sum(count == 0 for count in counts) / len(counts),
            2,
        ),
    },
    "missing_information": missing_report,
    "topics": topic_report,
    "observed_retracted_records": sum(
        paper.get("observed_is_retracted") is True for paper in train
    ),
    "five_most_cited_training_papers": [
        {
            "openalex_id": paper["openalex_id"],
            "title": paper["title"],
            "five_year_citations": paper["five_year_citation_count"],
        }
        for paper in top_papers
    ],
}

output_file = dataset_folder / "training_exploration_report.json"
output_file.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\nReport saved: {output_file}")