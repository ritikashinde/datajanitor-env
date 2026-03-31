import pandas as pd
import numpy as np


def count_missing(df: pd.DataFrame) -> dict:
    return df.isnull().sum().to_dict()


def count_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def get_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(include=["object"]).columns.tolist()


def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_inconsistencies(df: pd.DataFrame) -> dict:
    inconsistencies = {}

    for col in get_categorical_columns(df):
        unique_vals = sorted([str(v) for v in df[col].dropna().unique()])

        canon_map = {}
        for v in unique_vals:
            canon = v.strip().lower()
            canon_map.setdefault(canon, []).append(v)

        if any(len(vals) > 1 for vals in canon_map.values()):
            inconsistencies[col] = unique_vals

    return inconsistencies


def numeric_outlier_counts(df: pd.DataFrame) -> dict:
    outliers = {}

    for col in get_numeric_columns(df):
        series = df[col].dropna()

        if len(series) < 4:
            outliers[col] = 0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outliers[col] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers[col] = int(((df[col] < lower) | (df[col] > upper)).sum())

    return outliers


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in get_categorical_columns(df):
        df[col] = df[col].apply(
            lambda x: x.strip().lower().capitalize() if isinstance(x, str) else x
        )

    return df


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in get_numeric_columns(df):
        series = df[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = df[(df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))]

    return df.reset_index(drop=True)


def fill_missing_mean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in get_numeric_columns(df):
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    return df


def fill_missing_mode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def component_scores(df: pd.DataFrame) -> dict:
    missing_total = sum(count_missing(df).values())
    duplicate_total = count_duplicates(df)
    category_total = len(categorical_inconsistencies(df))
    outlier_total = sum(numeric_outlier_counts(df).values())

    return {
        "missing_score": 1.0 / (1.0 + missing_total),
        "duplicate_score": 1.0 / (1.0 + duplicate_total),
        "category_score": 1.0 / (1.0 + category_total),
        "outlier_score": 1.0 / (1.0 + outlier_total),
    }


def quality_score(df: pd.DataFrame) -> float:
    scores = component_scores(df)

    score = (
        0.30 * scores["missing_score"]
        + 0.25 * scores["duplicate_score"]
        + 0.25 * scores["category_score"]
        + 0.20 * scores["outlier_score"]
    )

    return round(float(score), 4)


def summarize_dataset(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "missing_counts": count_missing(df),
        "duplicate_count": count_duplicates(df),
        "categorical_inconsistencies": categorical_inconsistencies(df),
        "numeric_outlier_counts": numeric_outlier_counts(df),
        "quality_score": quality_score(df),
        "preview": df.head(5).fillna("").to_dict(orient="records"),
    }