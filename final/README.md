# Machine Learning Final Project

This folder contains the final submission artifacts for the scholarship prediction project.

## Required submission files

- `report.pdf`
- `notebook.ipynb`
- `predictions.csv`
- `main.tex` is included as the LaTeX source used to build `report.pdf`.

## Recommended supporting files

- `README.md`
- `requirements.txt`
- `build_final_artifacts.py`

## Data used

- Train: `ml_dataset_v2_train.csv`
- Dev: `ml_dataset_v2_dev.csv`
- Hidden test: `hidden_test_data.csv`

The notebook expects the CSV files to be in the same folder as `notebook.ipynb`, matching the simple midterm style. If `hidden_test_data.csv` does not include `part_time_job_hours`, the workflow fills that missing feature with the final-train median.

For Google Colab, upload these files into the same working folder as the notebook:

- `ml_dataset_v2_train.csv`
- `ml_dataset_v2_dev.csv`
- `hidden_test_data.csv`

## Constraint

The models are trained with `numpy`. The workflow does not use `scikit-learn` training APIs.

## Selected model

- Model: `Logistic Regression`
- Accuracy: `0.9200`
- Precision: `1.0000`
- Recall: `0.8000`
- F1-score: `0.8889`

## How to reproduce

From the repository root:

```powershell
python final/build_final_artifacts.py
```

Or open `final/notebook.ipynb` and run all cells from top to bottom.

To rebuild the report from LaTeX, compile `final/main.tex` from inside the `final/` folder and use the generated PDF as `report.pdf`.

## Output format

`predictions.csv` has exactly:

```csv
id,label_pred
```

where `label_pred` is either `0` or `1`.
