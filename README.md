---
title: DataJanitor Environment
emoji: 🧹
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# DataJanitorEnv: RL Environment for Data Cleaning

## Overview
This project is a reinforcement learning-style environment for automated data cleaning.

An agent interacts with datasets using actions like:
- remove_duplicates
- fill_missing_values
- standardize_categories
- remove_outliers

The goal is to clean messy datasets step-by-step.

## Actions
- inspect_dataset
- remove_duplicates
- fill_missing_mean
- fill_missing_mode
- standardize_categories
- remove_outliers_iqr
- finish

## Reward System
- Rewards for fixing missing values, duplicates, outliers
- Penalty for useless actions
- Finish gives reward only if dataset is fully clean

## Results
easy: 1.0  
medium: 1.0  
hard: 1.0  

## Tech Stack
FastAPI, Pandas, NumPy, Docker

## API Endpoints
/reset  
/step  
/grader  
/state  
## 🔍 API Documentation

Interactive API documentation is available via FastAPI Swagger UI:

👉 https://ritikashinde-datajanitor-env.hf.space/docs

You can use this interface to:
- explore endpoints
- test API calls
- visualize request/response formats