# Load Type Classification — Assignment 2

## Overview
This project classifies electrical power consumption into three categories:

- `Light_Load`
- `Medium_Load`
- `Maximum_Load`

It includes exploratory data analysis (EDA), model training, validation, and final evaluation on a held-out test month.

## Project Structure

```text
assignment-2 1/
├── data/
│   └── load_data.csv
├── artifacts/
│   └── load_type_model.joblib
├── eda.ipynb
├── model.ipynb
├── predict_load_type.py
├── predictions.csv          # Generated when predictions are run
├── requirements.txt
└── README.md
```

## Setup

Run these commands from the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Conda, activate your Conda environment instead, then install the requirements.

## Generate Predictions

Ensure the trained model exists at `artifacts/load_type_model.joblib`, then run:

```bash
python predict_load_type.py data/load_data.csv
```

By default, the script writes `predictions.csv` beside `predict_load_type.py`.

To choose a different output path:

```bash
python predict_load_type.py data/load_data.csv output/predictions.csv
```

The parent folder for a custom output path must already exist. The output CSV contains the input columns plus a `Predicted_Load_Type` column.

## Train and Evaluate

Run the notebooks in this order:

1. `eda.ipynb` — explores the data and checks data quality.
2. `model.ipynb` — trains and compares models, performs validation, evaluates on the held-out test month, and saves the trained model artifact.

The assignment uses December 2018 as the final test month. Use the December test metrics reported in `model.ipynb` for the final performance report, rather than metrics calculated over predictions for the full dataset.

## Modeling Limitation

The selected model uses `day_med` and `day_p90`, which summarize activity over the full day. These features rely on measurements from later in the same day, so this model is intended for retrospective classification—not prediction at the beginning of the day or true real-time use.

For real-time prediction, redesign the feature pipeline to use only information available at prediction time (for example, timestamp features and properly lagged historical measurements), then retrain and reevaluate the model.

## Troubleshooting

### Missing feature columns
If you see `Input is missing required feature columns`, confirm that you are using the updated `predict_load_type.py` script and that the script and model artifact belong to the same project version.

### Model artifact not found
Confirm that `artifacts/load_type_model.joblib` exists relative to the project folder.

### Missing dependencies
Install dependencies with:

```bash
pip install -r requirements.txt
```
