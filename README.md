---
title: DataJanitor Environment
emoji: 🧹
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# 🧹 DataJanitorEnv: RL Environment for Data Cleaning

## 🚀 Overview
DataJanitorEnv is a reinforcement learning-style environment for automated data cleaning.

An agent interacts with a dataset through discrete actions such as:
- removing duplicates
- filling missing values
- standardizing categories
- removing outliers

The goal is to transform a noisy dataset into a clean, high-quality dataset.

---

## ⚙️ Environment Design

### 📥 Observation Space
Each step returns:
- number of rows and columns
- missing value counts
- duplicate count
- categorical inconsistencies
- outlier counts
- quality score
- preview of dataset

---

### 🎯 Action Space

Available actions:
- `inspect_dataset`
- `remove_duplicates`
- `fill_missing_mean`
- `fill_missing_mode`
- `standardize_categories`
- `remove_outliers_iqr`
- `finish`

---

### 🏆 Reward Design

Reward is computed based on **actual improvements**:

- Reduction in missing values
- Removal of duplicates
- Reduction in outliers
- Fixing categorical inconsistencies

Penalties:
- Small step penalty to discourage unnecessary actions
- Strong penalty for ineffective actions
- Negative reward for premature `finish`

Positive reward:
- `finish` gives reward only if dataset is fully clean

---

## 🤖 Baseline Agent

A rule-based agent is implemented that:
- inspects dataset
- applies cleaning actions based on state
- avoids premature termination

It achieves:
