import pandas as pd
from utils import quality_score, summarize_dataset


def grade_dataset(df: pd.DataFrame) -> dict:
    score = quality_score(df)
    summary = summarize_dataset(df)

    return {
        "score": round(score, 4),
        "passed": score >= 0.85,
        "summary": summary,
    }


def grade_task_result(task_id: str, df: pd.DataFrame) -> dict:
    result = grade_dataset(df)
    result["task_id"] = task_id
    return result