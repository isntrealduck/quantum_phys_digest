import json
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

tuning_run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output_folder = dataset_folder / "small_tuning" / tuning_run
output_folder.mkdir(parents=True, exist_ok=False)

experiments = [
    {
        "name": "random_forest_text_leaf2",
        "source": "train_random_forest_text.py",
        "original_model_folder": "random_forest_text",
        "old_setting": '"min_samples_leaf": 5,',
        "new_setting": '"min_samples_leaf": 2,',
        "arguments": [],
    },
    {
        "name": "random_forest_text_leaf10",
        "source": "train_random_forest_text.py",
        "original_model_folder": "random_forest_text",
        "old_setting": '"min_samples_leaf": 5,',
        "new_setting": '"min_samples_leaf": 10,',
        "arguments": [],
    },
    {
        "name": "random_forest_text_half_features",
        "source": "train_random_forest_text.py",
        "original_model_folder": "random_forest_text",
        "old_setting": '"min_samples_leaf": 5,',
        "new_setting": (
            '"min_samples_leaf": 5,\n'
            '    "max_features": 0.5,'
        ),
        "arguments": [],
    },
    {
        "name": "catboost_text_institution_depth4",
        "source": "train_catboost_text_institution.py",
        "original_model_folder": "catboost_text_institution",
        "old_setting": '"depth": 6,',
        "new_setting": '"depth": 4,',
        "arguments": ["20261011T181812227144Z"],
    },
    {
        "name": "catboost_text_institution_l2_10",
        "source": "train_catboost_text_institution.py",
        "original_model_folder": "catboost_text_institution",
        "old_setting": '"depth": 6,',
        "new_setting": (
            '"depth": 6,\n'
            '    "l2_leaf_reg": 10,'
        ),
        "arguments": ["20261011T181812227144Z"],
    },
    {
        "name": "catboost_text_institution_iterations250",
        "source": "train_catboost_text_institution.py",
        "original_model_folder": "catboost_text_institution",
        "old_setting": '"iterations": 500,',
        "new_setting": '"iterations": 250,',
        "arguments": ["20261011T181812227144Z"],
    },
]

results = []

for number, experiment in enumerate(experiments, start=1):
    name = experiment["name"]

    source = (script_folder / experiment["source"]).read_text(
        encoding="utf-8-sig"
    )

    # Require the expected setting so changes cannot silently fail.
    if source.count(experiment["old_setting"]) != 1:
        raise ValueError(f"Could not locate the setting exactly once: {name}")

    original_folder = json.dumps(experiment["original_model_folder"])

    if source.count(original_folder) != 2:
        raise ValueError(f"Unexpected output-folder definitions: {name}")

    changed = source.replace(
        experiment["old_setting"],
        experiment["new_setting"],
    ).replace(original_folder, json.dumps(name))

    generated_script = script_folder / f"train_{name}.py"
    generated_script.write_text(changed, encoding="utf-8")

    print(f"[{number}/{len(experiments)}] Training {name}...", flush=True)

    process = subprocess.run(
        [
            sys.executable,
            str(generated_script),
            *experiment["arguments"],
        ],
        cwd=project_folder,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    (output_folder / f"{name}.log").write_text(
        process.stdout + "\n" + process.stderr,
        encoding="utf-8",
    )

    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        raise SystemExit(
            f"Training failed: {name}. Earlier results remain saved."
        )

    match = re.search(r'"run_id":\s*"([^"]+)"', process.stdout)

    if match is None:
        raise ValueError(f"Could not find the model run ID: {name}")

    model_run = match.group(1)
    report_file = dataset_folder / name / model_run / "report.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))

    result = {
        "model_folder": name,
        "model_run": model_run,
        "changed_setting": experiment["new_setting"],
        "training_metrics": report["training_metrics"],
        "validation_metrics": report["validation_metrics"],
    }
    results.append(result)

    (output_folder / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2), flush=True)

print("\nEntries to add inside MODELS in evaluate_batch_ranking.py:")

for result in results:
    print(
        f'    "{result["model_folder"]}": "{result["model_run"]}",'
    )

print(f"\nTuning results: {output_folder / 'results.json'}")