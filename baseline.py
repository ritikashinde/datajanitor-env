import requests

BASE_URL = "http://127.0.0.1:8000"


def run_task(task_id: str):
    scores = {}

    reset_response = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    reset_response.raise_for_status()

    actions = [
        "inspect_dataset",
        "remove_duplicates",
        "fill_missing_mode",
        "standardize_categories",
        "remove_outliers_iqr",
        "finish",
    ]

    history = []

    for action in actions:
        step_response = requests.post(
            f"{BASE_URL}/step",
            json={"action": action}
        )
        step_response.raise_for_status()
        result = step_response.json()

        history.append({
            "action": action,
            "reward": result["reward"],
            "quality_score": result["observation"]["quality_score"],
            "done": result["done"],
        })

        if result["done"]:
            break

    grader_response = requests.get(f"{BASE_URL}/grader")
    grader_response.raise_for_status()
    final_result = grader_response.json()

    return {
        "task_id": task_id,
        "score": final_result["score"],
        "passed": final_result["passed"],
        "history": history,
        "summary": final_result["summary"],
    }


def run_baseline():
    task_ids = ["easy", "medium", "hard"]
    details = {}
    total_score = 0.0

    for task_id in task_ids:
        result = run_task(task_id)
        details[task_id] = result
        total_score += result["score"]

    average_score = round(total_score / len(task_ids), 4)

    return {
        "scores": {task_id: details[task_id]["score"] for task_id in task_ids},
        "average_score": average_score,
        "details": details,
    }


if __name__ == "__main__":
    result = run_baseline()
    print(result)