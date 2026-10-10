import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


def normalize_text(value):
    """Ignore capitalization, accents, and punctuation."""
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(
        character for character in text
        if not unicodedata.combining(character)
    )
    text = text.casefold()
    return " ".join(re.findall(r"\w+", text))


def normalize_authors(value):
    """Represent each author's name as sorted words."""
    names = set()

    for name in (value or "").split(";"):
        words = normalize_text(name).split()
        if words:
            names.add(" ".join(sorted(words)))

    return names


if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python validate_matches.py COLLECTION_FOLDER MATCH_FOLDER"
    )

project_folder = Path(__file__).resolve().parent

# Use the exact matching run we want to validate.
for folder_name in sys.argv[1:]:
    if Path(folder_name).name != folder_name or folder_name in {".", ".."}:
        raise ValueError("Supply folder names, not full paths.")

match_folder = (
    project_folder
    / "data"
    / "processed"
    / sys.argv[1]
    / "openalex"
    / sys.argv[2]
)

matches = json.loads(
    (match_folder / "matches.json").read_text(encoding="utf-8")
)

results = []

for match in matches:
    row = dict(match)
    reasons = []

    original_title = normalize_text(row["arxiv_title"])
    candidate_title = normalize_text(row["openalex_title"])

    title_similarity = SequenceMatcher(
        None, original_title, candidate_title, autojunk=False
    ).ratio()

    original_authors = normalize_authors(row["arxiv_authors"])
    candidate_authors = normalize_authors(row["openalex_authors"])

    shared_authors = original_authors & candidate_authors
    all_authors = original_authors | candidate_authors

    author_overlap = (
        len(shared_authors) / len(all_authors)
        if all_authors else 0.0
    )

    if row["status"] != "doi_candidate_pending_review":
        review_group = "unresolved"
        reasons.append(row["status"])
    else:
        if not original_title or not candidate_title:
            reasons.append("missing title")
        elif title_similarity < 0.90:
            reasons.append("title differs")

        if not original_authors or not candidate_authors:
            reasons.append("missing authors")
        elif original_authors != candidate_authors:
            reasons.append("author names differ")

        review_group = "needs_review" if reasons else "passes_checks"

    row["title_similarity"] = round(title_similarity, 3)
    row["author_overlap"] = round(author_overlap, 3)
    row["review_group"] = review_group
    row["review_reason"] = "; ".join(reasons)

    results.append(row)


def write_csv(filename, rows):
    # Keep headers even when the filtered list is empty.
    with (match_folder / filename).open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


if not results:
    raise SystemExit("The matching file contains no records.")

write_csv("validation_all.csv", results)

flagged = [
    row for row in results
    if row["review_group"] != "passes_checks"
]

write_csv("validation_flagged.csv", flagged)

counts = Counter(row["review_group"] for row in results)

print(f"Records checked: {len(results)}")
for group in ("passes_checks", "needs_review", "unresolved"):
    print(f"{group}: {counts[group]}")

print(f"\nReview file: {match_folder / 'validation_flagged.csv'}")