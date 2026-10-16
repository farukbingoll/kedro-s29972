# Iris Classification with Kedro

This project predicts the species of an iris flower from four measurements in
centimetres: sepal length, sepal width, petal length and petal width.
It uses a small decision tree and a Kedro pipeline with five nodes.

## Dataset

The raw CSV contains 150 flowers from the UCI Iris dataset and is included in
`data/01_raw/iris.csv`. The target has three classes: Iris-setosa,
Iris-versicolor and Iris-virginica.

- Public source: https://archive.ics.uci.edu/dataset/53/iris
- Download: https://archive.ics.uci.edu/static/public/53/iris.zip
- Citation: Fisher, R. (1936). Iris. UCI Machine Learning Repository.
  https://doi.org/10.24432/C56C76
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
  https://creativecommons.org/licenses/by/4.0/

The CSV was made from UCI's `iris.data` by adding a header and removing empty
lines. No download is required when running the project.

## Installation and run

Use Python 3.12. Open a terminal in this folder (the folder containing
`pyproject.toml`). On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\kedro.exe run
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/kedro run
```

With the environment activated, the equivalent commands are:

```bash
kedro run
kedro run --pipeline=classification
kedro run --env=base
```

The default and named pipelines run the same five nodes.

## Pipeline

1. `validate_raw_data`: checks the required columns and rejects an empty dataset.
2. `clean_iris_data`: converts measurements to numbers and removes missing,
   empty or non-positive values. Repeated measurements are kept because they
   can represent different flowers.
3. `split_iris_data`: makes a stratified 80/20 training/test split.
4. `train_decision_tree`: trains a tree using only the training set.
5. `evaluate_decision_tree`: predicts the held-out test set and saves accuracy,
   macro F1 and a confusion matrix.

The split has 120 training rows and 30 test rows. The test rows are not used
for fitting. Cleaning does not estimate statistics from the complete dataset.

## Catalog and outputs

| Dataset | Type | File |
| --- | --- | --- |
| iris_raw | pandas.CSVDataset | data/01_raw/iris.csv |
| iris_clean | pandas.CSVDataset | data/02_intermediate/iris_clean.csv |
| iris_model | pickle.PickleDataset | data/06_models/iris_model.pkl |
| iris_metrics | json.JSONDataset | data/08_reporting/metrics.json |

Other node outputs are in-memory datasets managed by Kedro. Node functions do
not open files or contain file paths. All file access is handled by the catalog.
Generated files are included as examples and are overwritten by the next run.

## Verified result

All five nodes complete successfully with the default parameters. Accuracy is
0.966667 (29/30 correct), and macro F1 is 0.966583. The confusion matrix in
`data/08_reporting/metrics.json` uses true classes as rows and predicted classes
as columns, in the listed label order. This result describes one small test set.

## Parameters and environments

`conf/base/parameters.yml` defines the feature columns, target column, test
ratio, random seed and tree depth. The defaults are 0.2, 42 and 3 respectively.
For example, change `max_depth` to compare a different tree.

Shared configuration is in `conf/base/`. Kedro's default `local` environment
can override it with files in `conf/local/`. Local overrides and credentials
are ignored by Git. No credentials are needed for this dataset.

## Reproducibility

Tested with Python 3.12.14, Kedro 1.0.0, kedro-datasets 8.1.0, pandas 2.2.3,
NumPy 2.3.5 and scikit-learn 1.8.0. Direct dependencies are pinned in
`requirements.txt`. `requirements-lock.txt` pins the transitive dependencies
from the clean test environment as well. To recreate it on Python 3.12, use
`pip install -r requirements-lock.txt`, then `pip install --no-deps -e .`.
The included raw data, fixed seed and fixed split settings make repeated runs
reproducible within this environment. No external service is required to run.

## Submission

Create a GitHub repository and upload this folder's contents, including `src`,
`conf`, `data`, this README and the dependency files. Include `.gitignore`.
Do not upload a virtual environment or `conf/local` overrides.
Submit the repository link in the assignment.
