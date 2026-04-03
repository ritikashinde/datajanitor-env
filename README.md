# DataJanitorEnv

DataJanitorEnv is a real-world OpenEnv environment for cleaning messy tabular datasets through sequential actions such as removing duplicates, filling missing values, standardizing categorical values, and handling outliers.

## Why this project
Data cleaning is a core part of real machine learning workflows. This environment simulates that process as an interactive decision-making task for agents.

## Features
- 3 task levels: easy, medium, hard
- Sequential cleaning actions
- Deterministic reward shaping
- Grader with quality score from 0.0 to 1.0
- Rule-based baseline agent
- FastAPI endpoints for environment interaction

## Available Actions
- inspect_dataset
- remove_duplicates
- fill_missing_mean
- fill_missing_mode
- standardize_categories
- remove_outliers_iqr
- finish

## API Endpoints
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grader`
- `GET /baseline`

## Task Design
### Easy
Duplicates, missing values, simple categorical inconsistencies

### Medium
Duplicates, missing values, category inconsistencies, numeric outliers

### Hard
Noisier dataset with multiple issues and a tighter action budget

## Grading
The environment uses a deterministic quality score based on:
- missing value reduction
- duplicate removal
- categorical consistency
- outlier reduction

A score of `0.85+` is considered passing.

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload