"""
baseline.py — rule-based baseline agent using the parameterized action schema.

The baseline is intentionally not optimal on 'hard' (demonstrates the gap
an RL agent needs to close).  It uses explicit params to show the schema in
action — e.g. gender mapping to fix M/F vs Male/Female, clip instead of
remove on hard (preserves rows).
"""

import requests

BASE_URL = "http://127.0.0.1:8000"


# Per-task action sequences — list of (action_type, params) tuples
TASK_SEQUENCES = {
    "easy": [
        ("inspect_dataset",        {}),
        ("remove_duplicates",      {"subset": [], "keep": "first"}),
        ("fill_missing_mode",      {"columns": []}),
        ("standardize_categories", {"strategy": "title_strip", "columns": []}),
        ("finish",                 {}),
    ],
    "medium": [
        ("inspect_dataset",        {}),
        ("remove_duplicates",      {"subset": [], "keep": "first"}),
        ("fill_missing_mode",      {"columns": []}),
        ("standardize_categories", {
            "strategy": "map",
            "mapping": {
                "gender": {
                    "M": "Male", "F": "Female",
                    "male": "Male", "female": "Female",
                    "Male": "Male", "Female": "Female",
                }
            }
        }),
        ("clip_outliers_iqr",      {"columns": [], "iqr_multiplier": 1.5}),
        ("finish",                 {}),
    ],
    "hard": [
        ("inspect_dataset",        {}),
        ("remove_duplicates",      {"subset": [], "keep": "first"}),
        ("standardize_categories", {
            "strategy": "map",
            "mapping": {
                "gender": {
                    "M": "Male", "F": "Female",
                    "male": "Male", "female": "Female",
                    " F": "Female", " male ": "Male",
                    "Male": "Male", "Female": "Female",
                }
            }
        }),
        ("clip_outliers_iqr",      {"columns": ["age", "income"], "iqr_multiplier": 1.5}),
        ("fill_missing_mean",      {"columns": ["income"]}),
        ("fill_missing_mode",      {"columns": ["gender"]}),
        ("finish",                 {}),
    ],
}


def run_task(task_id: str) -> dict:
    reset_resp = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    reset_resp.raise_for_status()

    history = []
    sequence = TASK_SEQUENCES.get(task_id, TASK_SEQUENCES["easy"])

    for action_type, params in sequence:
        step_resp = requests.post(
            f"{BASE_URL}/step",
            json={"action_type": action_type, "params": params},
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        history.append({
            "action_type": action_type,
            "params":      params,
            "reward":      result["reward"],
            "quality_score": result["observation"]["quality_score"],
            "done":        result["done"],
        })

        if result["done"]:
            break

    grader_resp = requests.get(f"{BASE_URL}/grader")
    grader_resp.raise_for_status()
    final = grader_resp.json()

    return {
        "task_id": task_id,
        "score":   final["score"],
        "passed":  final["passed"],
        "history": history,
        "summary": final["summary"],
    }


def run_baseline() -> dict:
    task_ids = ["easy", "medium", "hard"]
    details = {}
    total_score = 0.0

    for task_id in task_ids:
        result = run_task(task_id)
        details[task_id] = result
        total_score += result["score"]

    average_score = round(total_score / len(task_ids), 4)

    return {
        "scores":        {t: details[t]["score"] for t in task_ids},
        "average_score": average_score,
        "details":       details,
    }


if __name__ == "__main__":
    import json
    result = run_baseline()
    print(json.dumps(result, indent=2))