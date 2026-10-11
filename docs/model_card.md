# Quantum Physics Digest: model card

## Purpose
Rank quantum-physics papers by predicted five-year citation impact.
Citation impact is a proxy and does not measure scientific quality.

## Selected model
CatBoostRegressor, depth 4, 500 iterations, learning rate 0.05.
Model run: `20261011T184918333620Z`.
Inputs: 11 basic metadata features, 64 text components,
and 6 institution-history features.

## Data and evaluation
5,000 OpenAlex articles across five quantum topics.
Training: 2015–2017, 3,000 papers.
Validation: 2018, 1,000 papers.
Test: 2019, 1,000 papers.
Preprocessing was fitted on training data only.
Model selection used validation results; test results were assessed afterward.

## Test results
- MAE: 15.5242
- RMSE: 53.3437
- RMSLE: 1.0979
- Precision@10: 0.5052
- NDCG@10 with log citation gain: 0.7682
- Citation lift over random selection: 2.706

Ranking evaluation uses repeated random pools of 50 papers, selecting 10.
The same 1,000 papers are reused; the 400 pools are not independent observations.
This is not an evaluation on actual daily publication batches.

## Limitations
The model substantially underpredicts highly cited papers.
Test mean actual citations: 19.119.
Test mean predicted citations: 7.3783.
Metadata and historical links were retrieved retrospectively, rather than
from verified publication-time snapshots.
The chronological split is a retrospective evaluation, not a simulation
of a model deployable in 2019: training labels matured after that date.
Missing metadata, publication-date uncertainty, indexing coverage,
and changes in research trends may affect performance.
Citation forecasts should be presented as uncertain estimates.

## Reproducibility
The release contains the model, fitted imputer, text preprocessor,
feature order, hashes, test report, and an environment snapshot.
Saved validation predictions were reproduced without retraining.
New-paper feature construction and daily collection remain to be implemented.
