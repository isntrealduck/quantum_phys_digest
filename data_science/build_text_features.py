import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


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
dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
papers = json.loads(dataset_bytes)

papers_by_id = {
    paper["openalex_id"]: paper for paper in papers
}

basic_bytes = (
    dataset_folder / "basic_features" / "features.json"
).read_bytes()
basic_table = json.loads(basic_bytes)

splits = json.loads(
    (dataset_folder / "splits.json").read_text(encoding="utf-8")
)

if (
    dataset_hash != basic_table["dataset_sha256"]
    or dataset_hash != splits["dataset_sha256"]
):
    raise ValueError("Dataset, basic features, and splits differ.")

rows = basic_table["rows"]
row_ids = [row["openalex_id"] for row in rows]

if (
    len(papers_by_id) != len(papers)
    or len(row_ids) != len(set(row_ids))
    or set(row_ids) != set(papers_by_id)
):
    raise ValueError("Duplicate or mismatched paper IDs.")

split_by_id = {}

for split_name, ids in splits["ids_by_split"].items():
    for paper_id in ids:
        if paper_id in split_by_id:
            raise ValueError("Overlapping splits.")
        split_by_id[paper_id] = split_name

if set(split_by_id) != set(row_ids):
    raise ValueError("Splits do not match feature rows.")

for row in rows:
    if row["split"] != split_by_id[row["openalex_id"]]:
        raise ValueError("Incorrect split assignment.")


# Use titles and abstracts only.
# Missing abstracts contribute no abstract text.
texts = []

for row in rows:
    paper = papers_by_id[row["openalex_id"]]
    text = (
        (paper.get("title") or "")
        + "\n"
        + (paper.get("abstract") or "")
    )
    texts.append(text)

train_positions = [
    position
    for position, row in enumerate(rows)
    if row["split"] == "train"
]

train_texts = [texts[position] for position in train_positions]

vectorizer_settings = {
    "max_features": 10000,
    "min_df": 3,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "lowercase": True,
}

vectorizer = TfidfVectorizer(**vectorizer_settings)

print("Learning vocabulary and word weights from training papers...")
train_tfidf = vectorizer.fit_transform(train_texts)

component_count = min(
    64,
    train_tfidf.shape[0] - 1,
    train_tfidf.shape[1] - 1,
)

if component_count < 1:
    raise ValueError("Insufficient text for dimensionality reduction.")

svd = TruncatedSVD(
    n_components=component_count,
    n_iter=7,
    random_state=42,
)

print("Learning text components from training papers...")
svd.fit(train_tfidf)

# Apply the already fitted transformations to all rows.
# Validation/test text does not change vocabulary, weights, or SVD.
print("Transforming paper text...")
all_tfidf = vectorizer.transform(texts)
text_features = svd.transform(all_tfidf)

if not np.all(np.isfinite(text_features)):
    raise ValueError("Invalid text feature values.")

text_feature_names = [
    f"text_svd_{number:02d}"
    for number in range(component_count)
]

combined_names = (
    basic_table["feature_names"] + text_feature_names
)

if len(combined_names) != len(set(combined_names)):
    raise ValueError("Duplicate feature names.")

combined_rows = []

for position, row in enumerate(rows):
    features = dict(row["features"])

    for name, value in zip(
        text_feature_names, text_features[position]
    ):
        features[name] = float(value)

    combined_rows.append({
        "openalex_id": row["openalex_id"],
        "split": row["split"],
        "features": features,
        "target": row["target"],
    })


run_id = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%S%fZ"
)

output_folder = (
    dataset_folder / "basic_text_features" / run_id
)
output_folder.mkdir(parents=True, exist_ok=False)

model_folder = (
    project_folder / "models" / "text_preprocessor" / run_id
)
model_folder.mkdir(parents=True, exist_ok=False)

preprocessor_file = model_folder / "preprocessor.joblib"

joblib.dump(
    {
        "vectorizer": vectorizer,
        "svd": svd,
        "text_rule": "title + newline + abstract; missing text = empty",
        "text_feature_names": text_feature_names,
        "dataset_sha256": dataset_hash,
    },
    preprocessor_file,
)

output = {
    "dataset_sha256": dataset_hash,
    "basic_feature_file_sha256": hashlib.sha256(
        basic_bytes
    ).hexdigest(),
    "feature_names": combined_names,
    "target_name": "five_year_citation_count",
    "publication_time_features_verified": False,
    "text_preprocessor_file": preprocessor_file.relative_to(
        project_folder
    ).as_posix(),
    "text_preprocessor_sha256": hashlib.sha256(
        preprocessor_file.read_bytes()
    ).hexdigest(),
    "rows": combined_rows,
}

(output_folder / "features.json").write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

report = {
    "text_feature_run": run_id,
    "records": len(combined_rows),
    "preprocessing_fit_split": "train",
    "training_records": len(train_positions),
    "vocabulary_size": len(vectorizer.vocabulary_),
    "basic_feature_count": len(basic_table["feature_names"]),
    "text_feature_count": component_count,
    "total_feature_count": len(combined_names),
    "training_text_variance_retained": round(
        float(svd.explained_variance_ratio_.sum()), 4
    ),
    "records_with_no_known_vocabulary_terms": {
        split_name: sum(
            all_tfidf.getrow(position).nnz == 0
            for position, row in enumerate(rows)
            if row["split"] == split_name
        )
        for split_name in splits["ids_by_split"]
    },
    "versions": {
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    },
}

(output_folder / "report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))
print(f"\nFeatures: {output_folder / 'features.json'}")
print(f"Text preprocessor: {preprocessor_file}")