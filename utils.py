"""
utils.py — parameterized data-cleaning primitives for DataJanitorEnv.

Every public cleaning function now accepts explicit column lists and strategy
parameters so the agent can take surgical, targeted actions instead of
always running on the entire DataFrame.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Column introspection helpers
# ---------------------------------------------------------------------------

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _resolve_columns(df: pd.DataFrame, columns: List[str], dtype_filter: str) -> List[str]:
    """
    Return the effective column list.
    - If `columns` is non-empty, validate each name exists in df.
    - If empty, fall back to all columns matching dtype_filter
      ('numeric' | 'categorical' | 'all').
    """
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        return columns

    if dtype_filter == "numeric":
        return get_numeric_columns(df)
    if dtype_filter == "categorical":
        return get_categorical_columns(df)
    return df.columns.tolist()


# ---------------------------------------------------------------------------
# Counting / summarisation helpers (unchanged contracts)
# ---------------------------------------------------------------------------

def count_missing(df: pd.DataFrame) -> Dict[str, int]:
    return df.isnull().sum().to_dict()


def count_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def categorical_inconsistencies(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Returns columns where the same logical value appears in multiple surface
    forms (e.g. 'Male', 'male', ' male', 'M').
    Detection: strip+lower canonical form — if one canonical maps to >1 raw
    value, that column has inconsistencies.
    """
    inconsistencies: Dict[str, List[str]] = {}
    for col in get_categorical_columns(df):
        unique_vals = sorted([str(v) for v in df[col].dropna().unique()])
        canon_map: Dict[str, List[str]] = {}
        for v in unique_vals:
            canon = v.strip().lower()
            canon_map.setdefault(canon, []).append(v)
        if any(len(vals) > 1 for vals in canon_map.values()):
            inconsistencies[col] = unique_vals
    return inconsistencies


def numeric_outlier_counts(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> Dict[str, int]:
    outliers: Dict[str, int] = {}
    for col in get_numeric_columns(df):
        series = df[col].dropna()
        if len(series) < 4:
            outliers[col] = 0
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outliers[col] = 0
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outliers[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return outliers


# ---------------------------------------------------------------------------
# Cleaning primitives — parameterized
# ---------------------------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    subset : columns to consider for duplicate detection. None → all columns.
    keep   : 'first' | 'last' | 'none' (drop ALL duplicates).
    """
    subset = subset or None          # empty list → None → pandas uses all cols
    keep_val = False if keep == "none" else keep
    return df.drop_duplicates(subset=subset, keep=keep_val).reset_index(drop=True)


def fill_missing_mean(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fill NaN in numeric columns with the column mean."""
    df = df.copy()
    cols = _resolve_columns(df, columns or [], "numeric")
    for col in cols:
        if col in get_numeric_columns(df) and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    return df


def fill_missing_mode(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fill NaN in any-dtype columns with the column mode."""
    df = df.copy()
    cols = _resolve_columns(df, columns or [], "all")
    for col in cols:
        if df[col].isnull().any():
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
    return df


def fill_missing_constant(
    df: pd.DataFrame,
    value,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fill NaN in specified columns with a literal constant."""
    df = df.copy()
    cols = _resolve_columns(df, columns or [], "all")
    for col in cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(value)
    return df


def standardize_categories(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: str = "title_strip",
    mapping: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    columns  : categorical columns to process. None / [] → all object cols.
    strategy :
        'lower_strip'  — strip + lowercase          (' Male ' → 'male')
        'title_strip'  — strip + Title Case          (' male'  → 'Male')
        'upper_strip'  — strip + UPPERCASE           (' male'  → 'MALE')
        'map'          — explicit per-column value mapping
    mapping  : Dict[col → Dict[raw → canonical]].  Required for strategy='map'.
    """
    df = df.copy()
    cols = _resolve_columns(df, columns or [], "categorical")

    STRATEGIES = {
        "lower_strip":  lambda x: x.strip().lower()             if isinstance(x, str) else x,
        "title_strip":  lambda x: x.strip().title()             if isinstance(x, str) else x,
        "upper_strip":  lambda x: x.strip().upper()             if isinstance(x, str) else x,
    }

    if strategy == "map":
        if not mapping:
            raise ValueError("'mapping' dict is required when strategy='map'")
        for col in cols:
            if col in mapping:
                df[col] = df[col].map(lambda x: mapping[col].get(x, x))
    else:
        fn = STRATEGIES.get(strategy)
        if fn is None:
            raise ValueError(f"Unknown strategy: '{strategy}'")
        for col in cols:
            df[col] = df[col].apply(fn)

    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Drop rows where any value in `columns` falls outside
    [Q1 - multiplier×IQR, Q3 + multiplier×IQR].
    """
    df = df.copy()
    cols = _resolve_columns(df, columns or [], "numeric")

    for col in cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        df = df[(df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))]

    return df.reset_index(drop=True)


def clip_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Clip values to the IQR fence instead of dropping rows.
    Preserves row count — safer when dataset is small.
    """
    df = df.copy()
    cols = _resolve_columns(df, columns or [], "numeric")

    for col in cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


# ---------------------------------------------------------------------------
# Quality scoring (weights unchanged)
# ---------------------------------------------------------------------------

def component_scores(df: pd.DataFrame) -> Dict[str, float]:
    missing_total   = sum(count_missing(df).values())
    duplicate_total = count_duplicates(df)
    category_total  = len(categorical_inconsistencies(df))
    outlier_total   = sum(numeric_outlier_counts(df).values())

    return {
        "missing_score":   1.0 / (1.0 + missing_total),
        "duplicate_score": 1.0 / (1.0 + duplicate_total),
        "category_score":  1.0 / (1.0 + category_total),
        "outlier_score":   1.0 / (1.0 + outlier_total),
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
        "rows":                       len(df),
        "columns":                    df.columns.tolist(),
        "missing_counts":             count_missing(df),
        "duplicate_count":            count_duplicates(df),
        "categorical_inconsistencies": categorical_inconsistencies(df),
        "numeric_outlier_counts":     numeric_outlier_counts(df),
        "quality_score":              quality_score(df),
        "preview":                    df.head(5).fillna("").to_dict(orient="records"),
    }