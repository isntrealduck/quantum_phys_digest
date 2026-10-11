import itertools
import json
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent

dataset_folder = (
    project_folder
    / "data"
    / "processed"
    / "openalex_historical"
    / "20261010T202611867850Z"
    / "labeled"
    / "20261010T204724059047Z"
)

FEATURE_RUN = "20261011T181812227144Z"
BASELINE_NAME = "catboost_text_institution_depth4"
BASELINE_RUN = "20261011T184918333620Z"

search_run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output_folder = dataset_folder / "random_search" / search_run
output_folder.mkdir(parents=True, exist_ok=False)

# Sample eight distinct combinations reproducibly.
combinations = list(itertools.product(
    [3, 4, 5, 6],          # depth
    [0.03, 0.05, 0.08],   # learning rate
    [250, 500, 750],      # iterations
    [3, 10, 20],          # L2 regularization
))
selected = random.Random(42).sample(combinations, 8)

source = (
    script_folder / "train_catboost_text_institution.py"
).read_text(encoding="utf-8-sig")

models = {BASELINE_NAME: BASELINE_RUN}
results = []


def run_script(path, arguments, log_name):
    process = subprocess.run(
        [sys.executable, str(path), *arguments],
        cwd=project_folder,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    (output_folder / log_name).write_text(
        process.stdout + "\n" + process.stderr,
        encoding="utf-8",
    )

    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        raise SystemExit("An experiment failed. Completed results remain saved.")

    return process.stdout


for number, (depth, learning_rate, iterations, l2) in enumerate(
    selected, start=1
):
    name = f"catboost_search_{search_run}_{number:02d}"

    replacements = {
        '"depth": 6,': f'"depth": {depth},',
        '"learning_rate": 0.05,': f'"learning_rate": {learning_rate},',
        '"iterations": 500,': f'"iterations": {iterations},',
        '"loss_function": "RMSE",': (
            '"loss_function": "RMSE",\n'
            f'    "l2_leaf_reg": {l2},'
        ),
    }

    changed = source

    for old, new in replacements.items():
        if changed.count(old) != 1:
            raise ValueError(f"Could not locate setting exactly once: {old}")
        changed = changed.replace(old, new)

    if changed.count('"catboost_text_institution"') != 2:
        raise ValueError("Unexpected experiment-folder definitions.")

    changed = changed.replace('"catboost_text_institution"', json.dumps(name))

    # Keep generated scripts next to the originals so root paths still work.
    script_path = script_folder / f"train_{name}.py"
    script_path.write_text(changed, encoding="utf-8")

    settings = {
        "depth": depth,
        "learning_rate": learning_rate,
        "iterations": iterations,
        "l2_leaf_reg": l2,
    }

    print(
        f"[{number}/8] Training {settings}",
        flush=True,
    )

    stdout = run_script(
        script_path, [FEATURE_RUN], f"trial_{number:02d}.log"
    )

    match = re.search(r'"run_id":\s*"([^"]+)"', stdout)
    if match is None:
        raise ValueError("Could not find the experiment run ID.")

    model_run = match.group(1)
    models[name] = model_run

    report = json.loads(
        (dataset_folder / name / model_run / "report.json").read_text(
            encoding="utf-8"
        )
    )

    results.append({
        "model_folder": name,
        "model_run": model_run,
        "settings": settings,
        "validation_metrics": report["validation_metrics"],
    })

    (output_folder / "trials.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print(json.dumps(report["validation_metrics"]), flush=True)


# Reuse the existing evaluator, including its identical batches and checks.
evaluation_source = (
    script_folder / "evaluate_batch_ranking.py"
).read_text(encoding="utf-8-sig")

evaluation_source, replacements_made = re.subn(
    r"^MODELS\s*=\s*\{.*?^\}",
    "MODELS = " + json.dumps(models, indent=4),
    evaluation_source,
    count=1,
    flags=re.MULTILINE | re.DOTALL,
)

if replacements_made != 1:
    raise ValueError("Could not replace the evaluator's MODELS dictionary.")

evaluation_script = script_folder / f"evaluate_search_{search_run}.py"
evaluation_script.write_text(evaluation_source, encoding="utf-8")

print("\nEvaluating validation ranking...", flush=True)
stdout = run_script(evaluation_script, [], "ranking.log")

match = re.search(r"^Full report:\s*(.+)$", stdout, flags=re.MULTILINE)
if match is None:
    raise ValueError("Could not locate the ranking report.")

ranking_path = Path(match.group(1).strip())
ranking_report = json.loads(ranking_path.read_text(encoding="utf-8"))
summaries = ranking_report["summaries"]

baseline_report = json.loads(
    (
        dataset_folder / BASELINE_NAME / BASELINE_RUN / "report.json"
    ).read_text(encoding="utf-8")
)

all_results = [{
    "model_folder": BASELINE_NAME,
    "model_run": BASELINE_RUN,
    "settings": baseline_report["model_settings"],
    "validation_metrics": baseline_report["validation_metrics"],
}, *results]

for result in all_results:
    result["ranking_metrics"] = summaries[result["model_folder"]]

# Primary criterion: validation NDCG. MAE breaks exact ties.
ordered = sorted(
    all_results,
    key=lambda result: (
        -result["ranking_metrics"]["mean_ndcg_at_10_log_citation_gain"],
        result["validation_metrics"]["mae"],
        result["model_folder"],
    ),
)

summary = {
    "search_run": search_run,
    "random_trials": 8,
    "selection_metric": "Validation NDCG@10 with log citation gain",
    "test_set_used": False,
    "ranking_report": str(ranking_path),
    "winner": ordered[0],
    "comparison": [
        {
            "model_folder": result["model_folder"],
            "model_run": result["model_run"],
            "mae": result["validation_metrics"]["mae"],
            "rmsle": result["validation_metrics"]["rmsle"],
            "ndcg": result["ranking_metrics"][
                "mean_ndcg_at_10_log_citation_gain"
            ],
            "precision_at_10": result["ranking_metrics"][
                "mean_precision_at_10"
            ],
        }
        for result in ordered
    ],
}

(output_folder / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print(json.dumps(summary, indent=2))
print(f"\nSearch results: {output_folder / 'summary.json'}")