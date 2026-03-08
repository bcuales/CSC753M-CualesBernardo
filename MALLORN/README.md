# CSC751M — Kaggle: MALLORN Astronomical Classification Challenge

This folder contains a lightweight solution script for the Kaggle competition **MALLORN Astronomical Classification Challenge**.

## What’s included

- `mallorn_improved.py` — feature extraction + model training + submission generation
- `submissions/submission.csv` — an upload-ready Kaggle submission file

Large Kaggle data files are intentionally excluded from git via `.gitignore`.

## How to run (locally)

1. Download and unzip the competition data so you have the folder:
   `mallorn-astronomical-classification-challenge/` containing `train_log.csv`, `test_log.csv`, and `split_*/`.

2. From this directory:

```bash
python3 mallorn_improved.py
```

It will write `submission.csv` in the current folder.

## How to submit

Upload the generated CSV to Kaggle. The required format is:

- columns: `object_id,prediction`
- `prediction` must be `0` or `1`

You can also upload the provided final file:

- `submissions/submission.csv`
